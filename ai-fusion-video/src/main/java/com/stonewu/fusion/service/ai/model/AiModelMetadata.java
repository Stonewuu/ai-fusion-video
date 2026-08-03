package com.stonewu.fusion.service.ai.model;

/**
 * AI 模型显式元数据。
 */
public record AiModelMetadata(
        String platform,
        String normalizedPlatform,
        String modelProtocol
) {
}
