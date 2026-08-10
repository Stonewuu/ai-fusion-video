package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.common.BusinessException;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 提供商注册表，按平台标识解析具体实现。
 */
@Component
public class AiProviderRegistry {

    private final List<AiProvider> providers;

    public AiProviderRegistry(List<AiProvider> providers) {
        this.providers = providers;
    }

    public AiProvider getProvider(AiProviderContext context) {
        String platform = context.getPlatform();
        return providers.stream()
                .filter(provider -> provider.supports(platform))
                .findFirst()
                .orElseThrow(() -> new BusinessException("不支持的模型平台: " + platform));
    }
}