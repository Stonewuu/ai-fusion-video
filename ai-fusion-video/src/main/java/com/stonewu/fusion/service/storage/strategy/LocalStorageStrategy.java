package com.stonewu.fusion.service.storage.strategy;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageStrategy;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.CopyOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.TimeUnit;

/**
 * 本地文件存储策略
 * <p>
 * 将远程文件下载到本地磁盘，通过 Spring MVC 静态资源映射提供 HTTP 访问。
 */
@Component
@Slf4j
public class LocalStorageStrategy implements StorageStrategy {

    private static final String DEFAULT_BASE_PATH = "./data/media";
    private static final String URL_PREFIX = "/media";

    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(60, TimeUnit.SECONDS)
            .followRedirects(true)
            .build();

    @Override
    public String getType() {
        return "local";
    }

    @Override
    public String store(String remoteUrl, String subDir, StorageConfig config) {
        String basePath = resolveBasePath(config);

        Request request = new Request.Builder().url(remoteUrl).build();
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                throw new RuntimeException("下载文件失败: HTTP " + response.code() + " url=" + remoteUrl);
            }

            String extension = guessExtension(remoteUrl, response.header("Content-Type"));
            String filename = IdUtil.fastSimpleUUID() + "." + extension;

            Path dir = Paths.get(basePath, subDir);
            Files.createDirectories(dir);
            Path target = dir.resolve(filename);

            try (InputStream is = response.body().byteStream()) {
                Files.copy(is, target, StandardCopyOption.REPLACE_EXISTING);
            }

            long fileSize = Files.size(target);
            log.info("[LocalStorage] 文件已保存: {} ({} bytes)", target, fileSize);

            return URL_PREFIX + "/" + subDir + "/" + filename;
        } catch (IOException e) {
            throw new RuntimeException("本地存储文件失败: " + e.getMessage(), e);
        }
    }

    @Override
    public String storeBytes(byte[] data, String subDir, String extension, StorageConfig config) {
        String basePath = resolveBasePath(config);
        String filename = IdUtil.fastSimpleUUID() + "." + extension;

        try {
            Path dir = Paths.get(basePath, subDir);
            Files.createDirectories(dir);
            Path target = dir.resolve(filename);
            Files.write(target, data);

            log.info("[LocalStorage] 文件已保存: {} ({} bytes)", target, data.length);
            return URL_PREFIX + "/" + subDir + "/" + filename;
        } catch (IOException e) {
            throw new RuntimeException("本地存储文件失败: " + e.getMessage(), e);
        }
    }

    @Override
    public String storeFile(Path filePath, String subDir, String extension, StorageConfig config) {
        String basePath = resolveBasePath(config);
        String filename = IdUtil.fastSimpleUUID() + "." + extension;

        try {
            Path dir = Paths.get(basePath, subDir);
            Files.createDirectories(dir);
            Path target = dir.resolve(filename);
            Files.copy(filePath, target, new CopyOption[]{StandardCopyOption.REPLACE_EXISTING});

            long fileSize = Files.size(target);
            log.info("[LocalStorage] 文件已保存: {} ({} bytes)", target, fileSize);
            return URL_PREFIX + "/" + subDir + "/" + filename;
        } catch (IOException e) {
            throw new RuntimeException("本地存储文件失败: " + e.getMessage(), e);
        }
    }

    private String resolveBasePath(StorageConfig config) {
        if (config != null && StrUtil.isNotBlank(config.getBasePath())) {
            return config.getBasePath();
        }
        return DEFAULT_BASE_PATH;
    }

    /**
     * 推断文件扩展名
     */
    private String guessExtension(String url, String contentType) {
        // 优先从 Content-Type 推断
        if (StrUtil.isNotBlank(contentType)) {
            String lower = contentType.toLowerCase();
            if (lower.contains("png")) return "png";
            if (lower.contains("jpeg") || lower.contains("jpg")) return "jpg";
            if (lower.contains("gif")) return "gif";
            if (lower.contains("webp")) return "webp";
            if (lower.contains("svg")) return "svg";
            if (lower.contains("mp4")) return "mp4";
            if (lower.contains("webm")) return "webm";
            if (lower.contains("mov") || lower.contains("quicktime")) return "mov";
        }

        // 从 URL 提取
        try {
            String path = url.split("\\?")[0]; // 去掉 query 参数
            int dotIdx = path.lastIndexOf('.');
            if (dotIdx > 0) {
                String ext = path.substring(dotIdx + 1).toLowerCase();
                if (ext.length() <= 5) {
                    return ext;
                }
            }
        } catch (Exception ignored) {
        }

        return "bin";
    }
}
