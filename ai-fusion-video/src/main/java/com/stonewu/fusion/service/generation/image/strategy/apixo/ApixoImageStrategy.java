package com.stonewu.fusion.service.generation.image.strategy.apixo;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.apixo.ApixoGenerationRequests;
import com.stonewu.fusion.service.ai.apixo.ApixoTasks;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import org.springframework.stereotype.Component;

@Component
public class ApixoImageStrategy implements ImageGenerationStrategy {
    @Override
    public String getName() {
        return "apixo";
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig config) {
        return ApixoTasks.prepare(config, model, ApixoGenerationRequests.image(task, model));
    }

    @Override
    public RemoteGenerationResult query(String id, ImageTask task, AiModel model, ApiConfig config) {
        return ApixoTasks.query(config, model, id);
    }
}
