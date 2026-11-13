package com.stonewu.fusion.service.generation.image.strategy.youkou;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.youkou.YoukouContract;
import com.stonewu.fusion.service.ai.youkou.YoukouGenerationRequests;
import com.stonewu.fusion.service.ai.youkou.YoukouHttpClient;
import com.stonewu.fusion.service.ai.youkou.YoukouTasks;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class YoukouImageStrategy implements ImageGenerationStrategy {
    private final YoukouGenerationRequests requests;

    @Override
    public String getName() {
        return "youkou";
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig config) {
        return requests.image(task, model, config);
    }

    @Override
    public RemoteGenerationResult query(String id, ImageTask task, AiModel model, ApiConfig config) {
        return YoukouTasks.query(new YoukouHttpClient(config), id, YoukouContract.from(model));
    }
}
