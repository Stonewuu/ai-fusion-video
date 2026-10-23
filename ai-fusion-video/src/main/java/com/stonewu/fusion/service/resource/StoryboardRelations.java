package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.support.SFunction;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardRelations {
    private final StoryboardEpisodeMapper episodes;
    private final ResourceReadRepository resources;

    public void validate(ResourceHeader board, CanvasValueType type, ObjectNode fields, Long excludingId) {
        if (type == STORYBOARD_EPISODE) {
            if (fields.hasNonNull("episodeNumber")) {
                requireUnique(StoryboardEpisode::getEpisodeNumber, fields.path("episodeNumber").intValue(),
                        board.resourceId(),
                        excludingId, "集号");
            }
            if (fields.hasNonNull("scriptEpisodeId")) {
                long id = StoryboardFields.id(fields.get("scriptEpisodeId"), "scriptEpisodeId");
                var scriptEpisode = resources.require(SCRIPT_EPISODE, id, true);
                var source = resources.data(board);
                if (!Objects.equals(scriptEpisode.projectId(), board.projectId())
                        || !source.hasNonNull("scriptId")
                        || source.path("scriptId").longValue() != resources.data(scriptEpisode).path("scriptId")
                        .longValue()) {
                    throw StoryboardFields.invalid("剧本分集不属于当前分镜关联的剧本");
                }
                requireUnique(StoryboardEpisode::getScriptEpisodeId, id, board.resourceId(), excludingId,
                        "剧本分集绑定");
            }
        } else if (type == SHOT) {
            if (fields.hasNonNull("sceneAssetItemId")) {
                requireAssetItem(board.projectId(), fields.path("sceneAssetItemId").longValue(), "scene");
            }
            for (String key : List.of("characterIds", "propIds")) {
                if (fields.hasNonNull(key)) {
                    for (JsonNode id : fields.get(key)) {
                        requireAssetItem(board.projectId(), id.longValue(),
                                key.equals("characterIds") ? "character" : "prop");
                    }
                }
            }
        }
    }

    private void requireUnique(SFunction<StoryboardEpisode, ?> column, Object value, long board, Long excluding,
                               String label) {
        long count = episodes.selectCount(new LambdaQueryWrapper<StoryboardEpisode>()
                .eq(StoryboardEpisode::getStoryboardId, board).eq(column, value)
                .ne(excluding != null, StoryboardEpisode::getId, excluding));
        if (count > 0) {
            throw StoryboardFields.invalid(label + " 已存在，请使用实际分镜集 ID 修改或先解除绑定");
        }
    }

    private void requireAssetItem(long project, long id, String type) {
        var item = resources.require(ASSET_ITEM, id, true);
        var asset = resources.require(ASSET, resources.data(item).path("assetId").longValue(), true);
        if (!Objects.equals(asset.projectId(), project) || !type.equals(
                resources.data(asset).path("type").textValue())) {
            throw StoryboardFields.invalid("镜头资产引用必须属于当前项目且类型正确");
        }
    }
}
