package com.stonewu.fusion.service.ai.catalog;

import java.util.List;

/** 平台身份及配置模板；专用协议限定提供商，但不自动推断模型的请求协议。 */
public record ProviderDefinition(
        String id,
        String platform,
        String label,
        String url,
        String textProtocol,
        String imageProtocol,
        String videoProtocol,
        boolean appendV1,
        String vendor,
        String hint,
        boolean sponsored,
        boolean hidden,
        boolean quickSetup,
        String credentialUrl,
        String documentationUrl,
        List<Integer> supportedModelTypes,
        String discovery,
        List<String> modelPresetCodes,
        List<String> restrictedProtocols) {
    public ProviderDefinition {
        restrictedProtocols = restrictedProtocols == null ? List.of() : List.copyOf(restrictedProtocols);
    }
}
