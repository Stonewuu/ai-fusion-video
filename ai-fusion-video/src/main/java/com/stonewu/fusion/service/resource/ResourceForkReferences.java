package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 只处理业务协议中声明的编号；普通正文和自定义属性中的数字不作猜测性替换。
 */
public final class ResourceForkReferences {
    private ResourceForkReferences() {
    }

    public record Key(CanvasValueType type, long id) {
    }

    public static List<Key> references(CanvasValueType type, ObjectNode source) {
        List<Key> result = new ArrayList<>();
        rewrite(type, source, (target, id) -> {
            result.add(new Key(target, id));
            return id;
        });
        return List.copyOf(result);
    }

    public static ObjectNode rewrite(CanvasValueType type, ObjectNode source,
                                     BiFunction<CanvasValueType, Long, Long> ids) {
        ObjectNode data = source.deepCopy();
        switch (type) {
            case PROJECT -> {
            }
            case SCRIPT -> {
                field(data, "projectId", PROJECT, ids);
                for (var character : data.path("charactersJson")) {
                    if (character.isObject()) {
                        field((ObjectNode) character, "assetId", ASSET, ids);
                    }
                }
            }
            case SCRIPT_EPISODE -> field(data, "scriptId", SCRIPT, ids);
            case SCRIPT_SCENE -> {
                field(data, "scriptId", SCRIPT, ids);
                field(data, "episodeId", SCRIPT_EPISODE, ids);
                field(data, "sceneAssetId", ASSET, ids);
                array(data, "characterAssetIds", ASSET, ids);
                array(data, "propAssetIds", ASSET, ids);
                for (var dialogue : data.path("dialogues")) {
                    if (dialogue.isObject()) {
                        field((ObjectNode) dialogue, "character_asset_id", ASSET, ids);
                    }
                }
            }
            case STORYBOARD -> {
                field(data, "projectId", PROJECT, ids);
                field(data, "scriptId", SCRIPT, ids);
            }
            case STORYBOARD_EPISODE -> {
                field(data, "storyboardId", STORYBOARD, ids);
                field(data, "scriptEpisodeId", SCRIPT_EPISODE, ids);
            }
            case STORYBOARD_SCENE -> {
                field(data, "storyboardId", STORYBOARD, ids);
                field(data, "episodeId", STORYBOARD_EPISODE, ids);
            }
            case SHOT -> {
                field(data, "storyboardId", STORYBOARD, ids);
                field(data, "storyboardEpisodeId", STORYBOARD_EPISODE, ids);
                field(data, "storyboardSceneId", STORYBOARD_SCENE, ids);
                field(data, "sceneAssetItemId", ASSET_ITEM, ids);
                array(data, "characterIds", ASSET_ITEM, ids);
                array(data, "propIds", ASSET_ITEM, ids);
            }
            case ASSET -> field(data, "projectId", PROJECT, ids);
            case ASSET_ITEM -> field(data, "assetId", ASSET, ids);
            default -> throw new BusinessException(400, "不能复制此业务类型");
        }
        return data;
    }

    private static void field(ObjectNode data, String name, CanvasValueType type,
                              BiFunction<CanvasValueType, Long, Long> ids) {
        var value = data.get(name);
        if (value == null || value.isNull()) {
            return;
        }
        data.put(name, ids.apply(type, id(value)));
    }

    private static void array(ObjectNode data, String name, CanvasValueType type,
                              BiFunction<CanvasValueType, Long, Long> ids) {
        var value = data.get(name);
        if (value == null || value.isNull()) {
            return;
        }
        if (!value.isArray()) {
            throw new BusinessException(409, "复制来源的编号列表无效：" + name);
        }
        var mapped = data.arrayNode();
        for (var item : value) {
            mapped.add(LongNode.valueOf(ids.apply(type, id(item))));
        }
        data.set(name, mapped);
    }

    private static long id(JsonNode value) {
        if (!value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() <= 0) {
            throw new BusinessException(409, "复制来源包含无效的业务引用");
        }
        return value.longValue();
    }
}
