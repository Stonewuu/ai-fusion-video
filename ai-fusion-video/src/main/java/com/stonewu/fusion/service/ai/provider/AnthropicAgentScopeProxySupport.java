package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.AnthropicChatModel;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.transport.ProxyConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;

/**
 * Anthropic AgentScope 代理兼容层。
 * <p>
 * 优先尝试调用未来 AgentScope 官方统一的 builder.proxy(ProxyConfig)；
 * 当前版本若尚未提供，则回退到本地最小兼容实现，便于后续删除。
 */
final class AnthropicAgentScopeProxySupport {

    private static final Logger log = LoggerFactory.getLogger(AnthropicAgentScopeProxySupport.class);

    private AnthropicAgentScopeProxySupport() {
    }

    static Model create(AiProviderContext context, GenerateOptions defaultOptions, String rootBaseUrl) {
        AnthropicChatModel.Builder builder = AnthropicChatModel.builder()
                .apiKey(context.getApiKey())
                .modelName(context.getModelName())
                .stream(true);
        if (defaultOptions != null) {
            builder.defaultOptions(defaultOptions);
        }
        if (StrUtil.isNotBlank(rootBaseUrl)) {
            builder.baseUrl(rootBaseUrl);
        }

        if (tryApplyOfficialProxy(builder, context.getApiConfig())) {
            return builder.build();
        }
        if (!AiProxySupport.isEnabled(context.getApiConfig())) {
            return builder.build();
        }
        return new ProxyAwareAnthropicChatModel(
                rootBaseUrl,
                context.getApiKey(),
                context.getModelName(),
                true,
                defaultOptions,
                null,
                context.getApiConfig());
    }

    private static boolean tryApplyOfficialProxy(AnthropicChatModel.Builder builder, ApiConfig apiConfig) {
        ProxyConfig proxyConfig = AiProxySupport.agentScopeProxyConfig(apiConfig);
        if (proxyConfig == null) {
            return false;
        }
        try {
            Method proxyMethod = builder.getClass().getMethod("proxy", ProxyConfig.class);
            proxyMethod.invoke(builder, proxyConfig);
            log.debug("[Anthropic] 使用 AgentScope 官方 builder.proxy(ProxyConfig) 配置代理");
            return true;
        } catch (NoSuchMethodException ignored) {
            return false;
        } catch (ReflectiveOperationException e) {
            throw new BusinessException("Anthropic AgentScope 代理配置失败: " + e.getMessage());
        }
    }
}