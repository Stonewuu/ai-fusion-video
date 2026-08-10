package com.stonewu.fusion.service.system;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

/**
 * 统一处理预设画风参考图的 URL 识别、API 路径映射与 classpath 资源读取。
 */
@Component
public class PresetArtStyleResourceResolver {

    private static final String LOCAL_ART_STYLE_PREFIX = "/art-styles/";
    private static final String PRESET_ART_STYLE_API_PREFIX = "/api/art-styles/";
    private static final String PRESET_ART_STYLE_RESOURCE_BASE = "static/art-styles/";

    public boolean isPresetArtStylePath(String sourceUrl) {
        if (StrUtil.isBlank(sourceUrl)) {
            return false;
        }
        String normalized = sourceUrl.trim();
        return normalized.startsWith(LOCAL_ART_STYLE_PREFIX)
                || normalized.startsWith(PRESET_ART_STYLE_API_PREFIX);
    }

    public String toApiPath(String sourceUrl) {
        String fileName = extractFileName(sourceUrl);
        if (StrUtil.isBlank(fileName)) {
            return null;
        }
        return PRESET_ART_STYLE_API_PREFIX + fileName;
    }

    public PresetArtStyleResource load(String sourceUrl) throws IOException {
        String fileName = extractFileName(sourceUrl);
        if (StrUtil.isBlank(fileName)) {
            throw new BusinessException("预设参考图路径非法: " + sourceUrl);
        }
        ClassPathResource resource = new ClassPathResource(PRESET_ART_STYLE_RESOURCE_BASE + fileName);
        if (!resource.exists() || !resource.isReadable()) {
            throw new BusinessException("预设参考图不存在: " + sourceUrl);
        }
        try (InputStream inputStream = resource.getInputStream()) {
            return new PresetArtStyleResource(fileName, inputStream.readAllBytes(), resolveMimeType(fileName));
        }
    }

    private String extractFileName(String sourceUrl) {
        if (StrUtil.isBlank(sourceUrl)) {
            return null;
        }
        String normalized = sourceUrl.trim();
        String fileName;
        if (normalized.startsWith(PRESET_ART_STYLE_API_PREFIX)) {
            fileName = normalized.substring(PRESET_ART_STYLE_API_PREFIX.length());
        } else if (normalized.startsWith(LOCAL_ART_STYLE_PREFIX)) {
            fileName = normalized.substring(LOCAL_ART_STYLE_PREFIX.length());
        } else {
            return null;
        }
        return sanitizeFileName(fileName);
    }

    private String sanitizeFileName(String fileName) {
        if (StrUtil.isBlank(fileName)) {
            return null;
        }
        String normalized = fileName;
        int queryIndex = normalized.indexOf('?');
        if (queryIndex >= 0) {
            normalized = normalized.substring(0, queryIndex);
        }
        int fragmentIndex = normalized.indexOf('#');
        if (fragmentIndex >= 0) {
            normalized = normalized.substring(0, fragmentIndex);
        }
        if (normalized.contains("..") || normalized.contains("/") || normalized.contains("\\")) {
            return null;
        }
        return normalized;
    }

    private String resolveMimeType(String sourceUrl) {
        String lower = sourceUrl.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return "image/jpeg";
        }
        if (lower.endsWith(".webp")) {
            return "image/webp";
        }
        if (lower.endsWith(".gif")) {
            return "image/gif";
        }
        return "image/png";
    }

    public record PresetArtStyleResource(String fileName, byte[] bytes, String mimeType) {
    }
}