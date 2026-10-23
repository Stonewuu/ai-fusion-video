package com.stonewu.fusion.service.canvas.capability;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.capability.CanvasConfigField.*;
import static com.stonewu.fusion.service.canvas.capability.CanvasDefinitions.*;

final class CanvasBusinessCapabilities {
    private CanvasBusinessCapabilities() {
    }

    static List<CanvasNodeCapability> definitions() {
        List<CanvasNodeCapability> result = new ArrayList<>();
        result.add(resource(CanvasNodeKind.PROJECT, "项目设定", CanvasValueType.PROJECT,
                port("project", CanvasValueType.PROJECT, false, false),
                port("style", CanvasValueType.TEXT, false, false), port("image", CanvasValueType.IMAGE, false, false)));
        result.add(resource(CanvasNodeKind.SCRIPT, "剧本", CanvasValueType.SCRIPT,
                port("script", CanvasValueType.SCRIPT, false, false),
                port("text", CanvasValueType.TEXT, false, false)));
        result.add(resource(CanvasNodeKind.SCRIPT_EPISODE, "剧本分集", CanvasValueType.SCRIPT_EPISODE,
                port("episode", CanvasValueType.SCRIPT_EPISODE, false, false),
                port("text", CanvasValueType.TEXT, false, false)));
        result.add(resource(CanvasNodeKind.SCRIPT_SCENE, "剧本场次", CanvasValueType.SCRIPT_SCENE,
                port("scene", CanvasValueType.SCRIPT_SCENE, false, false),
                port("text", CanvasValueType.TEXT, false, false)));
        result.add(resource(CanvasNodeKind.STORYBOARD, "分镜", CanvasValueType.STORYBOARD,
                port("storyboard", CanvasValueType.STORYBOARD, false, false)));
        result.add(resource(CanvasNodeKind.STORYBOARD_EPISODE, "分镜分集", CanvasValueType.STORYBOARD_EPISODE,
                port("episode", CanvasValueType.STORYBOARD_EPISODE, false, false),
                port("shots", CanvasValueType.SHOT, true, false)));
        result.add(resource(CanvasNodeKind.STORYBOARD_SCENE, "分镜场次", CanvasValueType.STORYBOARD_SCENE,
                port("scene", CanvasValueType.STORYBOARD_SCENE, false, false),
                port("shots", CanvasValueType.SHOT, true, false)));
        result.add(resource(CanvasNodeKind.SHOT, "镜头", CanvasValueType.SHOT,
                port("shot", CanvasValueType.SHOT, false, false), port("prompt", CanvasValueType.TEXT, false, false),
                port("firstFrame", CanvasValueType.IMAGE, false, false),
                port("lastFrame", CanvasValueType.IMAGE, false, false),
                port("video", CanvasValueType.VIDEO, false, false)));
        result.add(assetResource(CanvasNodeKind.ASSET, "资产", CanvasValueType.ASSET,
                port("asset", CanvasValueType.ASSET, false, false),
                port("items", CanvasValueType.ASSET_ITEM, true, false)));
        result.add(assetResource(CanvasNodeKind.ASSET_ITEM, "子资产", CanvasValueType.ASSET_ITEM,
                port("assetItem", CanvasValueType.ASSET_ITEM, false, false),
                port("image", CanvasValueType.IMAGE, false, false),
                port("video", CanvasValueType.VIDEO, false, false)));
        result.add(agent(CanvasNodeKind.SCRIPT_PARSE, "解析剧本", "script_full_parse",
                port("script", CanvasValueType.SCRIPT, false, true),
                port("script", CanvasValueType.SCRIPT, false, false)));
        result.add(agent(CanvasNodeKind.STORY_TO_SCRIPT, "故事转剧本", "story_to_script",
                port("story", CanvasValueType.TEXT, false, true),
                port("script", CanvasValueType.SCRIPT, false, false)));
        result.add(agent(CanvasNodeKind.EPISODE_PARSE, "解析分集", "script_episode_parse",
                port("episode", CanvasValueType.SCRIPT_EPISODE, false, true),
                port("scenes", CanvasValueType.SCRIPT_SCENE, true, false)));
        result.add(agent(CanvasNodeKind.STORYBOARD_GENERATE, "剧本转分镜", "script_to_storyboard",
                port("script", CanvasValueType.SCRIPT, false, true),
                port("storyboard", CanvasValueType.STORYBOARD, false, false)));
        result.add(agent(CanvasNodeKind.ASSET_EXTRACT, "提取资产", "canvas_asset_extractor",
                port("script", CanvasValueType.SCRIPT, false, true),
                port("assets", CanvasValueType.ASSET, true, false)));
        result.add(agent(CanvasNodeKind.ASSET_IMAGE_GENERATE, "生成资产图片", "asset_image_gen",
                port("assets", CanvasValueType.ASSET_ITEM, true, true),
                port("images", CanvasValueType.IMAGE, true, false)));
        result.add(capability(CanvasNodeKind.FRAME_GENERATE, "生成镜头首尾帧", "generation", "agent", true,
                fields(field("agentType", choice(true, "storyboard_frame_gen")),
                        field("modelId", integer(1, MAX_SAFE_ID, false)),
                        field("imageModelId", integer(1, MAX_SAFE_ID, false)),
                        field("maxImages", integer(0, 1000, false)),
                        field("maxAgentRuns", integer(1, 64, false)),
                        field("skillMode", choice(false, "INHERIT", "CUSTOM")), field("skillNames", stringList(8, 100)),
                        field("frameType", choice(true, "first", "last")), field("instruction", text(32000))),
                agentInputs(port("shots", CanvasValueType.SHOT, true, true)),
                List.of(port("images", CanvasValueType.IMAGE, true, false))));
        return result;
    }

    private static CanvasNodeCapability assetResource(CanvasNodeKind kind, String label, CanvasValueType type,
                                                      CanvasPortDefinition... outputs) {
        return new CanvasNodeCapability(kind, label, "resource", "resource", 1, false, type,
                Map.of(), List.of(), List.of(outputs));
    }

    private static CanvasNodeCapability agent(CanvasNodeKind kind, String label, String agentType,
                                              CanvasPortDefinition input, CanvasPortDefinition output) {
        Map<String, CanvasConfigField> fields = fields(field("agentType", choice(true, agentType)),
                field("modelId", integer(1, MAX_SAFE_ID, false)), field("imageModelId", integer(1, MAX_SAFE_ID, false)),
                field("videoModelId", integer(1, MAX_SAFE_ID, false)), field("maxImages", integer(0, 1000, false)),
                field("maxVideoSeconds", integer(0, 36000, false)),
                field("maxAgentRuns", integer(1, 64, false)),
                field("skillMode", choice(false, "INHERIT", "CUSTOM")), field("skillNames", stringList(8, 100)),
                field("instruction", text(32000)));
        return capability(kind, label, "creation", "agent", true, fields, agentInputs(input), List.of(output));
    }
}
