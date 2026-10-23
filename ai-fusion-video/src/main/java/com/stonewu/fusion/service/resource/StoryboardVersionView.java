package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 父集合版本从当前事务读取，不能存入只按子对象版本命中的正文缓存。
 */
@Component
@RequiredArgsConstructor
public class StoryboardVersionView {
    private final ResourceReadRepository resources;

    public ObjectNode attach(CanvasValueType type, ObjectNode data) {
        if (type == STORYBOARD) {
            return data;
        }
        data.put("storyboardVersion",
                resources.require(STORYBOARD, data.path("storyboardId").longValue(), false).version());
        if (type == STORYBOARD_SCENE) {
            data.put("episodeVersion",
                    resources.require(STORYBOARD_EPISODE, data.path("episodeId").longValue(), false).version());
        } else if (type == SHOT) {
            if (data.hasNonNull("storyboardEpisodeId")) {
                data.put("episodeVersion",
                        resources.require(STORYBOARD_EPISODE, data.path("storyboardEpisodeId").longValue(), false)
                                .version());
            }
            if (data.hasNonNull("storyboardSceneId")) {
                data.put("sceneVersion",
                        resources.require(STORYBOARD_SCENE, data.path("storyboardSceneId").longValue(), false)
                                .version());
            }
        }
        return data;
    }
}
