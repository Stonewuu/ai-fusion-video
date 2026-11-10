package com.stonewu.fusion.service.generation.image.strategy.gemini;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/** Gemini Developer API 与 Vertex 共用图片内容格式，各自使用对应地址和凭据。 */
@Component
@RequiredArgsConstructor
public class GeminiImageStrategy implements ImageGenerationStrategy {
    private final VertexGeminiImageProtocol protocol;

    @Override
    public String getName() {
        return "gemini";
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig) {
        return protocol.prepare(task, model, apiConfig, resolveDefaultSize(model, task));
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        throw new IllegalStateException("同步图片结果必须从已保存回执读取");
    }
}
