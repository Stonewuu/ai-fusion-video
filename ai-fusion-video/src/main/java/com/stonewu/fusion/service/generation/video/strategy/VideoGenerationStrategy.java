package com.stonewu.fusion.service.generation.video.strategy;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;

import java.util.Set;

/**
 * 供应商协议只做一次请求；重试、租约、额度和结果发布由持久化执行层负责。
 */
public interface VideoGenerationStrategy {
    String getName();

    default int batchSize(VideoTask task, AiModel model, ApiConfig apiConfig) {
        return 1;
    }

    default Set<String> getSupportedProtocols() {
        return Set.of(getName());
    }

    /**
     * 素材与请求体全部就绪后再返回，不发送生成请求。
     */
    PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig apiConfig);

    RemoteGenerationResult query(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig);

    /**
     * true 只代表已发出取消请求，不能据此提前释放模型额度。
     */
    default boolean cancel(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig) {
        return false;
    }
}
