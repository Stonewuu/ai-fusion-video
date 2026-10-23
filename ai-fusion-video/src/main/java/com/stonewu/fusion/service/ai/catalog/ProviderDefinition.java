package com.stonewu.fusion.service.ai.catalog;

import java.util.List;

/** 平台身份及配置模板；商业展示信息不参与模型协议路由。 */
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
        boolean quickSetup,
        String credentialUrl,
        String documentationUrl,
        List<Integer> supportedModelTypes,
        String discovery,
        List<String> modelPresetCodes) {
}
