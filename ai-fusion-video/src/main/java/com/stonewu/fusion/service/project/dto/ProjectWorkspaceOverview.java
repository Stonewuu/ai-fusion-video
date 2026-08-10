package com.stonewu.fusion.service.project.dto;

import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;

/** 项目固定工作区概览。 */
public record ProjectWorkspaceOverview(
        Script script,
        long scriptEpisodeCount,
        long scriptSceneCount,
        Storyboard storyboard,
        StoryboardStatistics storyboardStatistics) {
}
