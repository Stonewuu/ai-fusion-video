package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;

import java.util.Arrays;
import java.util.List;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 表单、计划和执行共用专业能力边界，避免选择不可能产生的输出后才付费失败。
 */
public final class CanvasAgentNodeContracts {
    private CanvasAgentNodeContracts() {
    }

    public record Profile(String type, List<CanvasValueType> inputTypes, boolean singleInput,
                          boolean requiresTypedInput, List<CanvasValueType> resultTypes, String hint) {
    }

    public static List<Profile> list() {
        return CanvasAgentNodeTypes.TYPES.stream().sorted().map(CanvasAgentNodeContracts::require).toList();
    }

    public static Profile require(String type) {
        return switch (type) {
            case "ai_media" -> profile(type, Arrays.asList(CanvasValueType.values()), false, false,
                    List.of(TEXT, SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE, ASSET, ASSET_ITEM, STORYBOARD,
                            STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT, IMAGE, VIDEO),
                    "连接要处理的素材或业务内容，并写清楚希望完成的修改。");
            case "concept_visualizer" ->
                    profile(type, Arrays.asList(CanvasValueType.values()), false, false, List.of(TEXT),
                            "填写画面构思，或连接素材进行分析；所选模型需要支持相应媒体输入。");
            case "story_to_script" -> profile(type, List.of(TEXT, SCRIPT), false, true,
                    List.of(SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE, ASSET, ASSET_ITEM, TEXT),
                    "连接故事文本或填写创作要求，内容会保存到当前项目剧本。");
            case "script_full_parse" -> profile(type, List.of(SCRIPT), true, true,
                    List.of(SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE, ASSET, ASSET_ITEM, TEXT),
                    "连接一个剧本，将原文解析为分集和场次。");
            case "script_episode_parse" -> profile(type, List.of(SCRIPT_EPISODE), true, true,
                    List.of(SCRIPT_SCENE, SCRIPT_EPISODE, SCRIPT, ASSET, ASSET_ITEM, TEXT),
                    "连接一个剧本分集，将原文解析为场次。");
            case "script_assistant" -> profile(type, List.of(SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE), false, true,
                    List.of(SCRIPT_SCENE, SCRIPT_EPISODE, SCRIPT, ASSET, ASSET_ITEM, TEXT),
                    "连接要修改的剧本、分集或场次，并填写修改要求。");
            case "episode_script_creator", "episode_scene_writer" -> profile(type, List.of(SCRIPT_EPISODE), true, true,
                    List.of(SCRIPT_SCENE, SCRIPT_EPISODE, SCRIPT, TEXT), "连接一个剧本分集，编写并保存该集场次与对白。");
            case "script_to_storyboard" -> profile(type, List.of(SCRIPT), true, true,
                    List.of(STORYBOARD, STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT, ASSET, ASSET_ITEM, TEXT),
                    "连接一个剧本，为当前项目创作分镜。");
            case "episode_storyboard_writer" -> profile(type, List.of(SCRIPT_EPISODE), true, true,
                    List.of(STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT, STORYBOARD, TEXT),
                    "连接一个剧本分集，为该集创作分镜。");
            case "storyboard_asset_preprocessor", "canvas_asset_extractor" ->
                    profile(type, List.of(SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE), false, true,
                            List.of(ASSET, ASSET_ITEM, TEXT), "连接剧本、分集或场次，提取资产或准备分镜需要的子资产。");
            case "asset_image_gen", "asset_image_executor" ->
                    profile(type, List.of(ASSET_ITEM), type.endsWith("executor"), true,
                            List.of(IMAGE, ASSET_ITEM, ASSET, TEXT),
                            "连接子资产，生成并保存图片；执行器每次处理一个子资产。");
            case "storyboard_frame_gen", "storyboard_frame_executor" ->
                    profile(type, List.of(SHOT), type.endsWith("executor"), true,
                            List.of(IMAGE, SHOT, STORYBOARD_SCENE, STORYBOARD_EPISODE, STORYBOARD, TEXT),
                            "连接镜头，生成并保存所选首帧或尾帧；执行器每次处理一个镜头。");
            case "storyboard_video_gen", "storyboard_video_executor" ->
                    profile(type, List.of(SHOT), type.endsWith("executor"), true,
                            List.of(VIDEO, SHOT, STORYBOARD_SCENE, STORYBOARD_EPISODE, STORYBOARD, TEXT),
                            "连接镜头，生成并保存视频；执行器每次处理一个镜头。");
            default -> throw CanvasRunException.invalid("专业智能体契约不存在");
        };
    }

    private static Profile profile(String type, List<CanvasValueType> inputs, boolean single, boolean required,
                                   List<CanvasValueType> outputs, String hint) {
        return new Profile(type, inputs, single, required, outputs, hint);
    }

    public static void configuration(CanvasNode node) {
        var profile = require(CanvasAgentNodeTypes.type(node));
        if ((node.kind() == CanvasNodeKind.ASSET_IMAGE_GENERATE || node.kind() == CanvasNodeKind.FRAME_GENERATE)
                && node.config().path("maxImages").asInt(0) == 0) {
            throw CanvasRunException.invalid("请开启图片生成并设置本次生成数量上限");
        }
        if (node.kind() != CanvasNodeKind.AGENT) {
            return;
        }
        if (!profile.inputTypes().contains(CanvasValueType.fromCode(node.config().path("inputType").asText()))) {
            throw CanvasRunException.invalid("所选输入类型不适用于此专业智能体。" + profile.hint());
        }
        if (!profile.resultTypes().contains(CanvasValueType.fromCode(node.config().path("resultType").asText()))) {
            throw CanvasRunException.invalid("此专业智能体不能产生所选类型的真实结果，请重新选择结果类型");
        }
        if (node.config().path("resultType").asText().equals("image") && node.config().path("maxImages")
                .asInt(0) == 0) {
            throw CanvasRunException.invalid("图片结果需要开启图片生成并设置数量上限");
        }
        if (node.config().path("resultType").asText().equals("video") && node.config().path("maxVideoSeconds")
                .asInt(0) == 0) {
            throw CanvasRunException.invalid("视频结果需要开启视频生成并设置时长上限");
        }
    }
}
