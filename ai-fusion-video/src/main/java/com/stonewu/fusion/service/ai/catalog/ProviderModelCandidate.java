package com.stonewu.fusion.service.ai.catalog;

import java.util.List;
import java.util.Map;

/** 仅返回模型配置所需字段，不把上游计费或渠道数据透传到管理界面。 */
public record ProviderModelCandidate(String id, String name, Integer modelType, String modelProtocol,
                                     String capabilityPresetCode, String config, String configurationHint,
                                     TextCapabilities textCapabilities) {
    public record TextCapabilities(List<String> inputTypes, Map<String, List<String>> inputTransports,
                                   Integer contextWindow, boolean reasoning, List<String> reasoningEfforts) {
    }
}
