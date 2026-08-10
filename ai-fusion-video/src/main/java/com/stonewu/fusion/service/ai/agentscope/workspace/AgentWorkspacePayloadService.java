package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.ResolvedS3StorageConfig;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.storage.StorageTypes;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import lombok.extern.slf4j.Slf4j;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDate;
import java.util.HexFormat;
import java.util.Objects;
import java.util.UUID;

@Component
@Slf4j
public class AgentWorkspacePayloadService {

    private static final String CONTENT_TYPE = "application/json; charset=utf-8";

    private final StorageConfigService storageConfigService;
    private final S3StorageConfigResolver s3ConfigResolver;
    private final S3ClientFactory s3ClientFactory;
    private final Path defaultLocalRoot;

    public AgentWorkspacePayloadService(
            StorageConfigService storageConfigService,
            S3StorageConfigResolver s3ConfigResolver,
            S3ClientFactory s3ClientFactory,
            @Value("${app.agent-workspace.local-base-path:./data/agent-workspace}") String localPath) {
        this.storageConfigService = Objects.requireNonNull(storageConfigService);
        this.s3ConfigResolver = Objects.requireNonNull(s3ConfigResolver);
        this.s3ClientFactory = Objects.requireNonNull(s3ClientFactory);
        this.defaultLocalRoot = Path.of(localPath).toAbsolutePath().normalize();
    }

    public AgentWorkspaceStoredPayload write(String payload, AgentWorkspaceLocation location) {
        byte[] bytes = payload.getBytes(StandardCharsets.UTF_8);
        String sha256 = sha256(bytes);
        return switch (location.backendType()) {
            case AgentWorkspaceBackend.DATABASE -> new AgentWorkspaceStoredPayload(
                    location.backendType(), null, null, null, payload, sha256, bytes.length);
            case AgentWorkspaceBackend.LOCAL -> writeLocal(bytes, location, sha256);
            case AgentWorkspaceBackend.OBJECT_STORAGE -> writeObject(bytes, location, sha256);
            default -> throw new IllegalStateException("Unsupported backend: " + location.backendType());
        };
    }

    public String read(AgentWorkspaceEntry entry) {
        return read(new AgentWorkspaceStoredPayload(
                entry.getBackendType(),
                entry.getStorageConfigId(),
                entry.getLocalPath(),
                entry.getContentRef(),
                entry.getPayload(),
                entry.getContentSha256(),
                entry.getContentSize() == null ? 0 : entry.getContentSize()));
    }

    public String read(AgentWorkspaceStoredPayload stored) {
        return switch (stored.backendType()) {
            case AgentWorkspaceBackend.DATABASE -> requirePayload(stored.databasePayload());
            case AgentWorkspaceBackend.LOCAL -> readLocal(stored);
            case AgentWorkspaceBackend.OBJECT_STORAGE -> readObject(stored);
            default -> throw new IllegalStateException("Unsupported backend: " + stored.backendType());
        };
    }

    public void delete(AgentWorkspaceStoredPayload stored) {
        if (stored == null || stored.contentRef() == null) {
            return;
        }
        try {
            if (AgentWorkspaceBackend.LOCAL.equals(stored.backendType())) {
                Files.deleteIfExists(resolveLocal(stored.localPath(), stored.contentRef()));
            } else if (AgentWorkspaceBackend.OBJECT_STORAGE.equals(stored.backendType())) {
                ResolvedS3StorageConfig config = resolveS3(stored.storageConfigId());
                s3ClientFactory.getClient(config).deleteObject(builder -> builder
                        .bucket(config.bucketName())
                        .key(stored.contentRef()));
            }
        } catch (Exception failure) {
            log.warn("清理智能体工作空间正文失败 [backend={}, ref={}]: {}",
                    stored.backendType(), stored.contentRef(), failure.getMessage());
        }
    }

    public void verify(AgentWorkspaceStoredPayload stored) {
        byte[] bytes = read(stored).getBytes(StandardCharsets.UTF_8);
        if (!stored.sha256().equals(sha256(bytes)) || stored.size() != bytes.length) {
            throw new BusinessException("智能体工作空间正文校验失败");
        }
    }

    public Path resolveLocalRoot(String configuredPath) {
        return configuredPath == null || configuredPath.isBlank()
                ? defaultLocalRoot
                : Path.of(configuredPath).toAbsolutePath().normalize();
    }

    private AgentWorkspaceStoredPayload writeLocal(
            byte[] bytes,
            AgentWorkspaceLocation location,
            String sha256) {
        Path root = resolveLocalRoot(location.localPath());
        String relative = objectKey("objects");
        Path target = resolveLocal(root.toString(), relative);
        try {
            Files.createDirectories(target.getParent());
            Path temporary = Files.createTempFile(target.getParent(), ".workspace-", ".tmp");
            Files.write(temporary, bytes);
            try {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException ignored) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
            }
            return new AgentWorkspaceStoredPayload(
                    AgentWorkspaceBackend.LOCAL,
                    null,
                    root.toString(),
                    relative,
                    null,
                    sha256,
                    bytes.length);
        } catch (IOException failure) {
            throw new BusinessException("写入本地智能体工作空间失败: " + failure.getMessage());
        }
    }

    private AgentWorkspaceStoredPayload writeObject(
            byte[] bytes,
            AgentWorkspaceLocation location,
            String sha256) {
        ResolvedS3StorageConfig config = resolveS3(location.storageConfigId());
        String relative = objectKey("agent-workspace/objects");
        String prefix = config.basePath() == null || config.basePath().isBlank()
                ? ""
                : config.basePath().replace('\\', '/').replaceAll("^/+|/+$", "") + "/";
        String key = prefix + relative;
        S3Client client = s3ClientFactory.getClient(config);
        client.putObject(PutObjectRequest.builder()
                        .bucket(config.bucketName())
                        .key(key)
                        .contentType(CONTENT_TYPE)
                        .contentLength((long) bytes.length)
                        .build(),
                RequestBody.fromBytes(bytes));
        return new AgentWorkspaceStoredPayload(
                AgentWorkspaceBackend.OBJECT_STORAGE,
                location.storageConfigId(),
                null,
                key,
                null,
                sha256,
                bytes.length);
    }

    private String readLocal(AgentWorkspaceStoredPayload stored) {
        try {
            return Files.readString(resolveLocal(stored.localPath(), stored.contentRef()), StandardCharsets.UTF_8);
        } catch (IOException failure) {
            throw new BusinessException("读取本地智能体工作空间失败: " + failure.getMessage());
        }
    }

    private String readObject(AgentWorkspaceStoredPayload stored) {
        ResolvedS3StorageConfig config = resolveS3(stored.storageConfigId());
        ResponseBytes<GetObjectResponse> response = s3ClientFactory.getClient(config).getObjectAsBytes(
                GetObjectRequest.builder()
                        .bucket(config.bucketName())
                        .key(stored.contentRef())
                        .build());
        return response.asUtf8String();
    }

    private ResolvedS3StorageConfig resolveS3(Long storageConfigId) {
        if (storageConfigId == null) {
            throw new BusinessException("对象存储模式必须选择存储配置");
        }
        StorageConfig storage = storageConfigService.getById(storageConfigId);
        if (!Integer.valueOf(1).equals(storage.getStatus()) || !StorageTypes.isS3Like(storage.getType())) {
            throw new BusinessException("智能体工作空间需要启用的 S3 兼容存储配置");
        }
        return s3ConfigResolver.resolve(storage);
    }

    private Path resolveLocal(String configuredRoot, String contentRef) {
        if (contentRef == null || contentRef.isBlank()) {
            throw new BusinessException("本地智能体工作空间正文引用为空");
        }
        Path root = resolveLocalRoot(configuredRoot);
        Path target = root.resolve(contentRef).normalize();
        if (!target.startsWith(root)) {
            throw new BusinessException("非法的智能体工作空间正文路径");
        }
        return target;
    }

    private String objectKey(String prefix) {
        LocalDate today = LocalDate.now();
        return prefix + "/" + today.getYear() + "/" + String.format("%02d", today.getMonthValue())
                + "/" + UUID.randomUUID().toString().replace("-", "") + ".json";
    }

    private String requirePayload(String payload) {
        if (payload == null) {
            throw new BusinessException("数据库中的智能体工作空间正文为空");
        }
        return payload;
    }

    public static String sha256(byte[] value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    public static String sha256(String value) {
        return sha256(value.getBytes(StandardCharsets.UTF_8));
    }
}
