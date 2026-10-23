package com.stonewu.fusion.service.canvas.capability;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.ArrayList;
import java.util.List;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 可由生成结果覆盖的明确字段；结构化对象编辑仍使用业务命令，不把模型文本当作任意 JSON 执行。
 */
public final class CanvasWritebackFields {
    public record Field(String key, String label, CanvasValueType targetType, CanvasValueType valueType, String path) {
    }

    private static final List<Field> FIELDS = definitions();

    private CanvasWritebackFields() {
    }

    public static List<Field> list() {
        return FIELDS;
    }

    public static Field require(ObjectNode config) {
        var field = FIELDS.stream().filter(item -> item.key().equals(config.path("binding").asText())).findFirst()
                .orElseThrow(() -> invalid("请选择要写入的业务字段"));
        if (!field.targetType().code().equals(config.path("targetType").asText())
                || !field.valueType().code().equals(config.path("valueType").asText())) {
            throw invalid("写入字段与目标或输入类型不匹配");
        }
        return field;
    }

    private static List<Field> definitions() {
        List<Field> fields = new ArrayList<>();
        add(fields, PROJECT, TEXT, "description", "项目介绍", "artStyleDescription", "画风描述", "artStyleImagePrompt",
                "画风提示词");
        add(fields, SCRIPT, TEXT, "rawContent", "剧本原文", "content", "剧本内容", "storySynopsis", "故事梗概",
                "summary", "剧本摘要", "genre", "题材", "targetAudience", "受众");
        add(fields, SCRIPT_EPISODE, TEXT, "title", "分集标题", "synopsis", "分集梗概", "rawContent", "分集原文");
        add(fields, SCRIPT_SCENE, TEXT, "sceneHeading", "场景标题", "location", "场景地点", "sceneDescription",
                "场景描述");
        add(fields, STORYBOARD, TEXT, "description", "分镜说明");
        add(fields, STORYBOARD_EPISODE, TEXT, "title", "分集标题", "synopsis", "分集梗概");
        add(fields, STORYBOARD_SCENE, TEXT, "sceneHeading", "场景标题", "location", "场景地点");
        add(fields, SHOT, TEXT, "content", "镜头内容", "sceneExpectation", "画面预期", "sound", "声音", "dialogue",
                "对白",
                "soundEffect", "音效描述", "music", "配乐描述", "remark", "备注", "firstFramePrompt", "首帧提示词",
                "lastFramePrompt", "尾帧提示词", "videoPrompt", "视频提示词");
        add(fields, ASSET, TEXT, "name", "资产名称", "description", "资产描述", "aiPrompt", "生成提示词");
        add(fields, ASSET_ITEM, TEXT, "name", "变体名称", "aiPrompt", "生成提示词");
        add(fields, PROJECT, IMAGE, "coverUrl", "项目封面", "artStyleImageUrl", "画风参考图");
        add(fields, ASSET, IMAGE, "coverUrl", "资产封面", "properties.thumbnailUrl", "缩略图");
        add(fields, ASSET_ITEM, IMAGE, "imageUrl", "变体图片", "thumbnailUrl", "变体缩略图");
        for (var type : List.of(ASSET, ASSET_ITEM)) {
            add(fields, type, VIDEO, "properties.videoUrl", "素材视频");
            add(fields, type, AUDIO, "properties.audioUrl", "素材音频");
        }
        add(fields, SHOT, IMAGE, "imageUrl", "镜头图片", "referenceImageUrl", "参考图片", "generatedImageUrl",
                "生成图片",
                "firstFrameImageUrl", "首帧图片", "lastFrameImageUrl", "尾帧图片", "customData.videoPosterUrl",
                "视频封面");
        add(fields, SHOT, VIDEO, "videoUrl", "镜头素材视频", "generatedVideoUrl", "生成视频");
        add(fields, STORYBOARD_EPISODE, VIDEO, "composedVideoUrl", "分集成片");
        add(fields, STORYBOARD_EPISODE, AUDIO, "bgmAudioUrl", "分集配乐", "bgmSfxAudioUrl", "分集音效");
        return List.copyOf(fields);
    }

    private static void add(List<Field> fields, CanvasValueType target, CanvasValueType value, String... names) {
        for (int i = 0; i < names.length; i += 2) {
            fields.add(new Field(target.code() + ":" + names[i], names[i + 1], target, value, names[i]));
        }
    }

    private static CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_CONFIG, message);
    }
}
