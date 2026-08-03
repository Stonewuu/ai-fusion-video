package com.stonewu.fusion.service.storyboard.dto;

/**
 * 分镜概览统计。
 *
 * @param episodeCount 分镜集数量
 * @param sceneCount   分镜场次数量
 * @param itemCount    分镜镜头数量
 */
public record StoryboardStatistics(
        long episodeCount,
        long sceneCount,
        long itemCount) {
}
