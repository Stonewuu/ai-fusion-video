package com.stonewu.fusion.service.system;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.system.SystemConfig;
import com.stonewu.fusion.mapper.system.SystemConfigMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

/**
 * 系统配置服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SystemConfigService {

    public static final String SITE_BASE_URL_KEY = "site_base_url";
    public static final String RESOURCE_BASE_URL_KEY = "resource_base_url";

    private final SystemConfigMapper systemConfigMapper;
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver;

    /**
     * 获取配置值
     */
    @Cacheable(value = "systemConfig", key = "#key", unless = "#result == null")
    public String getValue(String key) {
        SystemConfig config = systemConfigMapper.selectOne(
                new LambdaQueryWrapper<SystemConfig>()
                        .eq(SystemConfig::getConfigKey, key)
                        .last("LIMIT 1"));
        return config != null ? config.getConfigValue() : null;
    }

    /**
     * 设置配置值（不存在则创建）
     */
    @CacheEvict(value = "systemConfig", key = "#key")
    public void setValue(String key, String value) {
        String normalizedValue = normalizeConfigValue(key, value);
        SystemConfig existing = systemConfigMapper.selectOne(
                new LambdaQueryWrapper<SystemConfig>()
                        .eq(SystemConfig::getConfigKey, key)
                        .last("LIMIT 1"));
        if (existing != null) {
            existing.setConfigValue(normalizedValue);
            systemConfigMapper.updateById(existing);
        } else {
            SystemConfig config = SystemConfig.builder()
                    .configKey(key)
                    .configValue(normalizedValue)
                    .build();
            systemConfigMapper.insert(config);
        }
    }

    /**
     * 获取所有配置
     */
    public List<SystemConfig> getAll() {
        return systemConfigMapper.selectList(
                new LambdaQueryWrapper<SystemConfig>()
                        .orderByAsc(SystemConfig::getConfigKey));
    }

    /**
     * 获取站点访问域名
     */
    public String getSiteBaseUrl() {
        return normalizeBaseUrl(getValue(SITE_BASE_URL_KEY));
    }

    /**
     * 获取供外部服务访问后端资源的公网地址。
     */
    public String getPublicResourceBaseUrl() {
        return normalizeBaseUrl(getValue(RESOURCE_BASE_URL_KEY));
    }

    /**
     * 获取布尔配置值。
     */
    public boolean getBooleanValue(String key, boolean defaultValue) {
        String value = getValue(key);
        if (StrUtil.isBlank(value)) {
            return defaultValue;
        }
        return Boolean.parseBoolean(value);
    }

    /**
     * 是否允许公开注册。
     */
    public boolean isRegistrationEnabled() {
        return getBooleanValue("allow_register", false);
    }

    /**
     * 设置是否允许公开注册。
     */
    public void setRegistrationEnabled(boolean enabled) {
        setValue("allow_register", Boolean.toString(enabled));
    }

    /**
     * 将相对路径解析为完整的公网可访问 URL
     * <p>
     * 1. 已是完整 URL (http/https) → 直接返回
     * 2. 预设画风图 (/art-styles/** 或 /api/art-styles/**) 统一映射到后端静态资源端点 /api/art-styles/**
     * 3. 其他相对路径在有 resource_base_url 时直接拼接
     * 4. 没有资源公网地址时，相对路径无法解析为公网 URL，返回 null
     */
    public String resolvePublicUrl(String relativePath) {
        if (StrUtil.isBlank(relativePath)) {
            return null;
        }
        // 已经是完整 URL（如 OSS 直链）
        if (relativePath.startsWith("http://") || relativePath.startsWith("https://")) {
            return relativePath;
        }
        String normalizedPath = relativePath.startsWith("/") ? relativePath : "/" + relativePath;
        if (presetArtStyleResourceResolver.isPresetArtStylePath(normalizedPath)) {
            String apiPath = presetArtStyleResourceResolver.toApiPath(normalizedPath);
            if (StrUtil.isBlank(apiPath)) {
                return null;
            }
            String resourceBaseUrl = getPublicResourceBaseUrl();
            if (StrUtil.isBlank(resourceBaseUrl)) {
                return null;
            }
            return resourceBaseUrl + apiPath;
        }
        // 拼接后端资源公网地址
        String resourceBaseUrl = getPublicResourceBaseUrl();
        if (StrUtil.isNotBlank(resourceBaseUrl)) {
            return resourceBaseUrl + normalizedPath;
        }
        return null;
    }

    private String normalizeConfigValue(String key, String value) {
        if (!SITE_BASE_URL_KEY.equals(key) && !RESOURCE_BASE_URL_KEY.equals(key)) {
            return value;
        }
        String normalized = normalizeBaseUrl(value);
        if (StrUtil.isBlank(normalized)) {
            return "";
        }
        try {
            URI uri = new URI(normalized);
            boolean httpProtocol = "http".equalsIgnoreCase(uri.getScheme())
                    || "https".equalsIgnoreCase(uri.getScheme());
            if (!httpProtocol || StrUtil.isBlank(uri.getHost())) {
                throw new BusinessException("公网地址必须是完整的 http:// 或 https:// URL");
            }
            if (uri.getUserInfo() != null || uri.getQuery() != null || uri.getFragment() != null) {
                throw new BusinessException("公网地址不能包含认证信息、查询参数或锚点");
            }
            if (RESOURCE_BASE_URL_KEY.equals(key)
                    && uri.getPath() != null
                    && uri.getPath().replaceAll("/+$", "").endsWith("/api")) {
                throw new BusinessException("后端资源公网地址应填写根地址，不能包含末尾 /api");
            }
            return normalized;
        } catch (URISyntaxException e) {
            throw new BusinessException("公网地址格式不正确: " + value);
        }
    }

    private String normalizeBaseUrl(String value) {
        if (StrUtil.isBlank(value)) {
            return value;
        }
        return value.trim().replaceAll("/+$", "");
    }
}
