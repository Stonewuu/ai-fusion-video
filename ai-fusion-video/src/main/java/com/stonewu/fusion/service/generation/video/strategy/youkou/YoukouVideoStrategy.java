package com.stonewu.fusion.service.generation.video.strategy.youkou;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.youkou.YoukouContract;
import com.stonewu.fusion.service.ai.youkou.YoukouGenerationRequests;
import com.stonewu.fusion.service.ai.youkou.YoukouHttpClient;
import com.stonewu.fusion.service.ai.youkou.YoukouTasks;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class YoukouVideoStrategy implements VideoGenerationStrategy {
    private final YoukouGenerationRequests requests;

    @Override
    public String getName() {
        return "youkou";
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig config) {
        return requests.video(task, model, config);
    }

    @Override
    public RemoteGenerationResult query(String id, VideoTask task, AiModel model, ApiConfig config) {
        return YoukouTasks.query(new YoukouHttpClient(config), id, YoukouContract.from(model));
    }
}
