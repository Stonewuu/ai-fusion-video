package com.stonewu.fusion.service.ai.tool;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.script.ScriptService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Objects;

/**
 * Resolves AI tool resource ownership and verifies access against the current execution user.
 */
@Component
@RequiredArgsConstructor
public class ToolResourceAccessGuard {

    private static final String ACCESS_DENIED_MESSAGE = "资源不存在或无权访问";

    private final ProjectService projectService;
    private final ScriptService scriptService;
    private final StoryboardService storyboardService;

    public Project requireProject(Long projectId, Long userId) {
        Project project = projectService.getById(projectId);
        requireProjectAccess(project, userId);
        return project;
    }

    public Script requireScript(Long scriptId, Long userId) {
        Script script = scriptService.getById(scriptId);
        requireProject(script.getProjectId(), userId);
        return script;
    }

    public ScriptEpisode requireScriptEpisode(Long episodeId, Long userId) {
        ScriptEpisode episode = scriptService.getEpisodeById(episodeId);
        requireScript(episode.getScriptId(), userId);
        return episode;
    }

    public ScriptSceneItem requireScriptScene(Long sceneId, Long userId) {
        ScriptSceneItem scene = scriptService.getSceneById(sceneId);
        requireScript(scene.getScriptId(), userId);
        return scene;
    }

    public Storyboard requireStoryboard(Long storyboardId, Long userId) {
        Storyboard storyboard = storyboardService.getById(storyboardId);
        requireProject(storyboard.getProjectId(), userId);
        return storyboard;
    }

    public StoryboardEpisode requireStoryboardEpisode(Long episodeId, Long userId) {
        StoryboardEpisode episode = storyboardService.getEpisodeById(episodeId);
        requireStoryboard(episode.getStoryboardId(), userId);
        return episode;
    }

    public StoryboardEpisode requireStoryboardEpisode(
            Long storyboardId,
            Long episodeId,
            Long userId) {
        StoryboardEpisode episode = requireStoryboardEpisode(episodeId, userId);
        if (!Objects.equals(storyboardId, episode.getStoryboardId())) {
            throw new BusinessException("分镜集不属于指定分镜");
        }
        return episode;
    }

    public StoryboardScene requireStoryboardScene(Long sceneId, Long userId) {
        StoryboardScene scene = storyboardService.getSceneById(sceneId);
        requireStoryboard(scene.getStoryboardId(), userId);
        return scene;
    }

    public StoryboardItem requireStoryboardItem(Long itemId, Long userId) {
        StoryboardItem item = storyboardService.getItemById(itemId);
        requireStoryboard(item.getStoryboardId(), userId);
        return item;
    }

    private void requireProjectAccess(Project project, Long userId) {
        if (userId == null || !projectService.canAccessProject(project, userId)) {
            throw new BusinessException(ACCESS_DENIED_MESSAGE);
        }
    }
}
