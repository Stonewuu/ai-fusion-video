package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ApiConfigService {

    private final ApiConfigMapper apiConfigMapper;

    @Transactional
    public Long createApiConfig(ApiConfig apiConfig) {
        if (apiConfig.getAutoAppendV1Path() == null) {
            apiConfig.setAutoAppendV1Path(true);
        }
        apiConfig.setApiUrl(normalizeApiUrl(apiConfig.getPlatform(), apiConfig.getApiUrl()));
        normalizeProxyConfig(apiConfig);
        apiConfigMapper.insert(apiConfig);
        return apiConfig.getId();
    }

    @Transactional
    public void updateApiConfig(Long id, String name, String platform, String apiUrl,
                                 Boolean autoAppendV1Path,
                                 String proxyType, String proxyHost, Integer proxyPort,
                                 String proxyUsername, String proxyPassword,
                                 String apiKey, String appId, String appSecret,
                                 Long modelId, Integer status, String remark) {
        ApiConfig config = apiConfigMapper.selectById(id);
        if (config == null) throw new BusinessException(404, "API配置不存在");
        String effectivePlatform = platform != null ? platform : config.getPlatform();
        if (name != null) config.setName(name);
        if (platform != null) config.setPlatform(platform);
        if (apiUrl != null) config.setApiUrl(normalizeApiUrl(effectivePlatform, apiUrl));
        if (autoAppendV1Path != null) config.setAutoAppendV1Path(autoAppendV1Path);
        if (proxyType != null) config.setProxyType(proxyType);
        if (proxyHost != null) config.setProxyHost(proxyHost);
        if (proxyPort != null) config.setProxyPort(proxyPort);
        if (proxyUsername != null) config.setProxyUsername(proxyUsername);
        if (proxyPassword != null) config.setProxyPassword(proxyPassword);
        if (apiKey != null) config.setApiKey(apiKey);
        if (appId != null) config.setAppId(appId);
        if (appSecret != null) config.setAppSecret(appSecret);
        if (modelId != null) config.setModelId(modelId);
        if (status != null) config.setStatus(status);
        if (remark != null) config.setRemark(remark);
        normalizeProxyConfig(config);
        apiConfigMapper.updateById(config);
    }

    @Transactional
    public void deleteApiConfig(Long id) {
        apiConfigMapper.deleteById(id);
    }

    public ApiConfig getById(Long id) {
        return apiConfigMapper.selectById(id);
    }

    public PageResult<ApiConfig> getPage(String name, String platform, Integer status, int pageNo, int pageSize) {
        LambdaQueryWrapper<ApiConfig> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(name != null, ApiConfig::getName, name)
                .eq(platform != null, ApiConfig::getPlatform, platform)
                .eq(status != null, ApiConfig::getStatus, status)
                .orderByDesc(ApiConfig::getId);
        return PageResult.of(apiConfigMapper.selectPage(new Page<>(pageNo, pageSize), wrapper));
    }

    public List<ApiConfig> getEnabledList() {
        return apiConfigMapper.selectList(new LambdaQueryWrapper<ApiConfig>().eq(ApiConfig::getStatus, 1));
    }

    /**
     * 按平台标识获取启用的 API 配置列表
     */
    public List<ApiConfig> getListByPlatform(String platform) {
        return apiConfigMapper.selectList(new LambdaQueryWrapper<ApiConfig>()
                .eq(ApiConfig::getStatus, 1)
                .eq(ApiConfig::getPlatform, platform));
    }

    /**
     * 按多个平台标识获取启用的 API 配置列表
     */
    public List<ApiConfig> getListByPlatforms(List<String> platforms) {
        return apiConfigMapper.selectList(new LambdaQueryWrapper<ApiConfig>()
                .eq(ApiConfig::getStatus, 1)
                .in(ApiConfig::getPlatform, platforms));
    }

    private String normalizeApiUrl(String platform, String apiUrl) {
        if (StrUtil.isBlank(apiUrl)) {
            return null;
        }
        String normalizedApiUrl = apiUrl.trim();
        String defaultApiUrl = getPlatformDefaultApiUrl(platform);
        if (StrUtil.isNotBlank(defaultApiUrl) && isSameApiUrl(normalizedApiUrl, defaultApiUrl)) {
            return null;
        }
        return normalizedApiUrl;
    }

    private void normalizeProxyConfig(ApiConfig config) {
        String proxyType = AiProxySupport.normalizeProxyType(config.getProxyType());
        if (AiProxySupport.TYPE_NONE.equals(proxyType)) {
            config.setProxyType(null);
            config.setProxyHost(null);
            config.setProxyPort(null);
            config.setProxyUsername(null);
            config.setProxyPassword(null);
            return;
        }
        if (!AiProxySupport.isSupportedProxyType(proxyType)) {
            throw new BusinessException("不支持的代理类型: " + config.getProxyType());
        }
        if (StrUtil.isBlank(config.getProxyHost())) {
            throw new BusinessException("启用代理时代理主机不能为空");
        }
        Integer proxyPort = config.getProxyPort();
        if (proxyPort == null || proxyPort <= 0 || proxyPort > 65535) {
            throw new BusinessException("启用代理时代理端口必须在 1-65535 之间");
        }
        String proxyUsername = StrUtil.trim(config.getProxyUsername());
        if (StrUtil.isBlank(proxyUsername)) {
            if (StrUtil.isNotBlank(config.getProxyPassword())) {
                throw new BusinessException("启用代理认证时代理用户名不能为空");
            }
            config.setProxyUsername(null);
            config.setProxyPassword(null);
        } else {
            config.setProxyUsername(proxyUsername);
        }
        config.setProxyType(proxyType);
        config.setProxyHost(config.getProxyHost().trim());
    }

    private boolean isSameApiUrl(String currentApiUrl, String defaultApiUrl) {
        return normalizeComparableApiUrl(currentApiUrl)
                .equalsIgnoreCase(normalizeComparableApiUrl(defaultApiUrl));
    }

    private String normalizeComparableApiUrl(String apiUrl) {
        if (StrUtil.isBlank(apiUrl)) {
            return "";
        }
        return apiUrl.trim().replaceAll("/+$", "");
    }

    private String getPlatformDefaultApiUrl(String platform) {
        if (StrUtil.isBlank(platform)) {
            return null;
        }
        return switch (platform) {
            case "openai_compatible", "openai" -> "https://api.openai.com";
            case "volcengine" -> "https://ark.cn-beijing.volces.com";
            case "vertex_ai" -> "us-central1";
            case "GoogleFlowReverseApi" -> "http://localhost:8000";
            case "dashscope" -> "https://dashscope.aliyuncs.com";
            case "anthropic" -> "https://api.anthropic.com";
            case "ollama" -> "http://localhost:11434";
            default -> null;
        };
    }
}
