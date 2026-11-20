package com.stonewu.fusion.service.generation.video.strategy.apixo;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.apixo.ApixoGenerationRequests;
import com.stonewu.fusion.service.ai.apixo.ApixoTasks;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import org.springframework.stereotype.Component;

@Component
public class ApixoVideoStrategy implements VideoGenerationStrategy {
    @Override
    public String getName() {
        return "apixo";
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig config) {
        return ApixoTasks.prepare(config, model, ApixoGenerationRequests.video(task, model));
    }

    @Override
    public RemoteGenerationResult query(String id, VideoTask task, AiModel model, ApiConfig config) {
        return ApixoTasks.query(config, model, id);
    }
}
