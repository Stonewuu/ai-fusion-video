package com.stonewu.fusion.service.canvas.capability;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.*;

final class CanvasDefinitions {
    static final long MAX_SAFE_ID = 9_007_199_254_740_991L;

    private CanvasDefinitions() {
    }

    static CanvasNodeCapability capability(CanvasNodeKind kind, String label, String category,
                                           String executor, boolean project,
                                           Map<String, CanvasConfigField> fields,
                                           List<CanvasPortDefinition> inputs,
                                           List<CanvasPortDefinition> outputs) {
        return new CanvasNodeCapability(kind, label, category, executor, 1, project, null,
                fields, inputs, outputs);
    }

    static CanvasNodeCapability resource(CanvasNodeKind kind, String label, CanvasValueType type,
                                         CanvasPortDefinition... outputs) {
        return new CanvasNodeCapability(kind, label, "resource", "resource", 1, true, type,
                Map.of(), List.of(), List.of(outputs));
    }

    static CanvasPortDefinition port(String name, CanvasValueType type, boolean collection, boolean required) {
        return new CanvasPortDefinition(name, portLabel(name), type, null, collection, required, collection ? 256 : 1);
    }

    static CanvasPortDefinition typedPort(String name, String field, boolean collection, boolean required) {
        return new CanvasPortDefinition(name, portLabel(name), null, field, collection, required, collection ? 256 : 1);
    }

    static CanvasPortDefinition contextPort() {
        return new CanvasPortDefinition("context", "内容设定", null, null, true, false, 256,
                Arrays.stream(CanvasValueType.values())
                        .filter(type -> type == CanvasValueType.TEXT || type.isBusinessResource()).toList());
    }

    private static String portLabel(String name) {
        return switch (name) {
            case "prompt" -> "创作要求";
            case "context" -> "内容设定";
            case "text" -> "文本";
            case "texts" -> "文本集合";
            case "image" -> "图片";
            case "images" -> "图片集合";
            case "video" -> "视频";
            case "videos" -> "视频集合";
            case "firstFrame" -> "首帧";
            case "lastFrame" -> "尾帧";
            case "firstFrames" -> "首帧集合";
            case "lastFrames" -> "尾帧集合";
            case "referenceImages" -> "参考图片";
            case "referenceVideos" -> "参考视频";
            case "referenceAudios" -> "参考音频";
            case "referenceDocuments" -> "参考文档";
            case "audio" -> "音频";
            case "audios" -> "音频集合";
            case "media" -> "媒体";
            case "project" -> "项目";
            case "script" -> "剧本";
            case "storyboard" -> "分镜";
            case "episode" -> "分集";
            case "scene" -> "场次";
            case "scenes" -> "场次集合";
            case "shot" -> "镜头";
            case "shots" -> "镜头集合";
            case "asset" -> "资产";
            case "assets" -> "资产集合";
            case "assetItem" -> "子资产";
            case "items" -> "条目";
            case "style" -> "画风";
            case "story" -> "故事";
            case "music" -> "配乐";
            case "sfx" -> "音效";
            case "instruction" -> "创作要求";
            case "inputs" -> "输入集合";
            case "results" -> "输出集合";
            case "summary" -> "摘要";
            case "candidates" -> "候选集合";
            case "selected" -> "选定结果";
            case "input" -> "输入";
            case "output" -> "输出";
            case "target" -> "目标";
            case "value" -> "内容";
            default -> throw new IllegalArgumentException("未登记的端口名称: " + name);
        };
    }

    static List<CanvasPortDefinition> agentInputs(CanvasPortDefinition... primary) {
        List<CanvasPortDefinition> inputs = new ArrayList<>(List.of(primary));
        inputs.add(
                new CanvasPortDefinition("referenceImages", "参考图片", CanvasValueType.IMAGE, null, true, false, 32));
        inputs.add(
                new CanvasPortDefinition("referenceVideos", "参考视频", CanvasValueType.VIDEO, null, true, false, 32));
        inputs.add(
                new CanvasPortDefinition("referenceAudios", "参考音频", CanvasValueType.AUDIO, null, true, false, 32));
        inputs.add(
                new CanvasPortDefinition("referenceDocuments", "参考文档", CanvasValueType.DOCUMENT, null, true, false,
                        32));
        return List.copyOf(inputs);
    }

    static Map.Entry<String, CanvasConfigField> field(String name, CanvasConfigField definition) {
        return Map.entry(name, definition);
    }

    @SafeVarargs
    static Map<String, CanvasConfigField> fields(Map.Entry<String, CanvasConfigField>... definitions) {
        Map<String, CanvasConfigField> result = new LinkedHashMap<>();
        for (Map.Entry<String, CanvasConfigField> definition : definitions) {
            if (result.putIfAbsent(definition.getKey(), definition.getValue()) != null) {
                throw new IllegalArgumentException("重复的节点配置字段: " + definition.getKey());
            }
        }
        return result;
    }
}
