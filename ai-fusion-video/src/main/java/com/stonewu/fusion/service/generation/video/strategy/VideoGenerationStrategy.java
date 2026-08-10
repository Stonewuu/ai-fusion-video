package com.stonewu.fusion.service.generation.video.strategy;

import com.stonewu.fusion.entity.generation.VideoTask;

import java.util.Set;

/**
 * 视频生成策略接口
 * <p>
 * 不同平台实现此接口
 */
public interface VideoGenerationStrategy {

    /**
     * 策略名称
     */
    String getName();

    /**
     * 当前策略支持的显式协议键。大多数策略只有一个协议；聚合协议可暴露多个子协议。
     */
    default Set<String> getSupportedProtocols() {
        return Set.of(getName());
    }

    /**
     * 提交生视频任务到平台
     *
     * @param task 生视频任务
     * @return 平台任务ID
     */
    String submit(VideoTask task);

    /**
     * 轮询平台任务状态
     *
     * @param platformTaskId 平台任务ID
     * @param task 生视频任务
     */
    void poll(String platformTaskId, VideoTask task);

    /** Cancel a submitted remote task when the platform supports cancellation. */
    default boolean cancel(String platformTaskId, VideoTask task) {
        return false;
    }

    /** Whether result URLs have already been copied into platform-controlled storage. */
    default boolean persistsResults() {
        return false;
    }
}
