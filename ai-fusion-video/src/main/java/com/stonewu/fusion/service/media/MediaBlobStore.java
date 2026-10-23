package com.stonewu.fusion.service.media;

import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLocks;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLocks.Domain;
import com.stonewu.fusion.service.media.model.MediaBlob;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import com.stonewu.fusion.service.storage.StorageTypes;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.core.exception.SdkException;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.IOException;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Objects;
import java.util.function.BooleanSupplier;

@Component
@RequiredArgsConstructor
public class MediaBlobStore {
    private final StorageConfigMapper configurations;
    private final S3StorageConfigResolver s3Config;
    private final S3ClientFactory s3Clients;
    private final MediaCryptor cryptor;
    private final MediaBlobLocks locks;

    public long write(MediaBlob blob, Path plaintext) throws IOException {
        return write(blob, plaintext, Domain.MEDIA);
    }

    public long writeDelivery(MediaBlob blob, Path plaintext) throws IOException {
        return write(blob, plaintext, Domain.DELIVERY);
    }

    private long write(MediaBlob blob, Path plaintext, Domain domain) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        StorageConfig config = configuration(blob);
        return locks.exclusive(blob.id(), domain, state -> {
            if (!"STAGING".equals(state)) {
                throw new IOException("媒体写入代数已结束，不能重新写入旧对象");
            }
            return writeLocked(blob, plaintext, config);
        });
    }

    private long writeLocked(MediaBlob blob, Path plaintext, StorageConfig config) throws IOException {
        if ("local".equals(blob.backend())) {
            Path target = localPath(config, blob.objectKey(), true);
            try (var stream = Files.newOutputStream(target, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
                cryptor.encrypt(plaintext, stream, blob.keyset(), blob.resourceId());
            }
            return Files.size(target);
        }
        Path encrypted = Files.createTempFile("afv-ciphertext-", ".blob");
        try {
            try (var stream = Files.newOutputStream(encrypted)) {
                cryptor.encrypt(plaintext, stream, blob.keyset(), blob.resourceId());
            }
            var resolved = s3Config.resolve(config);
            long bytes = Files.size(encrypted);
            try {
                s3Clients.getClient(resolved).putObject(PutObjectRequest.builder().bucket(resolved.bucketName())
                        .key(s3Key(config, blob.objectKey())).contentLength(bytes)
                        .overrideConfiguration(request -> request.apiCallTimeout(Duration.ofMinutes(10)))
                        .contentType("application/octet-stream").build(), RequestBody.fromFile(encrypted));
            } catch (SdkException failure) {
                throw new IOException("媒体对象存储写入失败", failure);
            }
            return bytes;
        } finally {
            Files.deleteIfExists(encrypted);
        }
    }

    /**
     * 只由孤儿回收任务调用；对象锁内用短事务复核保留关系及租约，再在事务外删除。
     */
    public boolean purge(MediaBlob blob, BooleanSupplier allowed) throws IOException {
        return purge(blob, Domain.MEDIA, allowed);
    }

    public boolean purgeDelivery(MediaBlob blob, BooleanSupplier allowed) throws IOException {
        return purge(blob, Domain.DELIVERY, allowed);
    }

    private boolean purge(MediaBlob blob, Domain domain, BooleanSupplier allowed) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        StorageConfig config = configuration(blob);
        return locks.purge(blob.id(), domain, allowed, state -> {
            if ("local".equals(blob.backend())) {
                validateKey(blob.objectKey());
                if (config.getBasePath() == null || config.getBasePath().isBlank()) {
                    throw new IOException("本地存储根目录不可用");
                }
                Path base = Path.of(config.getBasePath()).toAbsolutePath().normalize();
                base.toRealPath();
                if (!Files.notExists(base.resolve(".afv-managed"), LinkOption.NOFOLLOW_LINKS)) {
                    Files.deleteIfExists(localPath(config, blob.objectKey(), false));
                }
            } else {
                var resolved = s3Config.resolve(config);
                try {
                    s3Clients.getClient(resolved).deleteObject(
                            request -> request.bucket(resolved.bucketName()).key(s3Key(config, blob.objectKey()))
                                    .overrideConfiguration(options -> options.apiCallTimeout(Duration.ofSeconds(30))));
                } catch (SdkException failure) {
                    throw new IOException("媒体对象存储清理失败", failure);
                }
            }
            return true;
        });
    }

    public SeekableByteChannel open(MediaBlob blob) throws IOException {
        if (blob.storedBytes() == null) {
            throw new IOException("媒体对象尚未写入完成");
        }
        StorageConfig config = configuration(blob);
        SeekableByteChannel ciphertext;
        if ("local".equals(blob.backend())) {
            Path path = localPath(config, blob.objectKey(), false);
            ciphertext = Files.newByteChannel(path, StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS);
            if (ciphertext.size() != blob.storedBytes()) {
                ciphertext.close();
                throw new IOException("媒体对象长度不符");
            }
        } else {
            var resolved = s3Config.resolve(config);
            ciphertext = new S3MediaReadChannel(s3Clients.getClient(resolved), resolved.bucketName(),
                    s3Key(config, blob.objectKey()), blob.storedBytes());
        }
        try {
            return cryptor.decrypt(ciphertext, blob.keyset(), blob.resourceId());
        } catch (IOException | RuntimeException failure) {
            ciphertext.close();
            throw failure;
        }
    }

    private StorageConfig configuration(MediaBlob blob) {
        // 密钥与配置不能从可回填旧值的内容缓存中取得。
        StorageConfig config = configurations.selectById(blob.storageConfigId());
        if (config == null || !Objects.equals(config.getStatus(), 1)
                || !blob.backend().equals(StorageTypes.normalizeType(config.getType()))) {
            throw MediaException.unavailable("媒体使用的存储配置不可用");
        }
        return config;
    }

    private Path localPath(StorageConfig config, String key, boolean create) throws IOException {
        validateKey(key);
        if (config.getBasePath() == null || config.getBasePath().isBlank()) {
            throw MediaException.unavailable("本地媒体存储必须明确配置根目录");
        }
        Path base = Path.of(config.getBasePath()).toAbsolutePath().normalize();
        if (create) {
            Files.createDirectories(base);
        }
        Path realBase = base.toRealPath();
        Path folder = base.resolve(".afv-managed");
        if (create) {
            Files.createDirectories(folder);
        }
        if (Files.isSymbolicLink(folder) || !folder.toRealPath().startsWith(realBase)) {
            throw new IOException("媒体存储目录不允许链接到根目录之外");
        }
        Path target = folder.resolve(key);
        if (Files.isSymbolicLink(target)) {
            throw new IOException("媒体对象不允许使用符号链接");
        }
        return target;
    }

    private String s3Key(StorageConfig config, String key) {
        validateKey(key);
        String base = config.getBasePath();
        if (base == null || base.isBlank()) {
            return ".afv-managed/" + key;
        }
        String prefix = base.replaceAll("^/+|/+$", "");
        if (prefix.contains("..") || prefix.contains("\\") || prefix.contains("\n") || prefix.contains("\r")) {
            throw MediaException.unavailable("媒体对象前缀无效");
        }
        return prefix + "/.afv-managed/" + key;
    }

    private void validateKey(String key) {
        if (!key.matches("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\.afvblob")) {
            throw new IllegalStateException("托管媒体定位数据无效");
        }
    }
}
