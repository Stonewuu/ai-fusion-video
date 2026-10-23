package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.capability.CanvasWritebackFields;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 审批展示完整的用户内容；内部引用经权限检查后转换为名称。
 */
@Component
@RequiredArgsConstructor
public class CanvasApprovalText {
    private final CanvasModelCatalog models;
    private final ResourceQueryService resources;
    private final MediaResourceQueries media;
    private final CanvasMcpContracts contracts;
    private final CanvasCapabilityRegistry capabilities;
    private static final Set<String> INTERNAL = Set.of("id", "version", "configVersion", "layoutVersion", "sortOrder",
            "contractVersion", "resourceVersion");
    private static final Map<String, String> LABELS = Map.ofEntries(
            Map.entry("EXT", "外景"),
            Map.entry("INT", "内景"),
            Map.entry("action", "操作"),
            Map.entry("add", "新增"),
            Map.entry("agentType", "智能体"),
            Map.entry("skillMode", "Skill 使用方式"),
            Map.entry("skillNames", "工作规范"),
            Map.entry("INHERIT", "跟随智能体"),
            Map.entry("CUSTOM", "单独指定"),
            Map.entry("ai", "AI 生成"),
            Map.entry("aiGenerated", "AI 生成"),
            Map.entry("aiPrompt", "生成提示词"),
            Map.entry("all", "全部"),
            Map.entry("apiConfigId", "音频服务"),
            Map.entry("appearance", "外观"),
            Map.entry("append", "追加"),
            Map.entry("argumentsJson", "工具输入内容"),
            Map.entry("artStyle", "画风预设"),
            Map.entry("artStyleDescription", "画风描述"),
            Map.entry("artStyleImagePrompt", "参考图提示词"),
            Map.entry("artStyleImageUrl", "画风参考图"),
            Map.entry("assetId", "资产"),
            Map.entry("assetIds", "资产"),
            Map.entry("assetItemId", "子资产"),
            Map.entry("assetName", "资产名称"),
            Map.entry("assetType", "资产类型"),
            Map.entry("assets", "资产"),
            Map.entry("atmosphere", "氛围"),
            Map.entry("back", "背面"),
            Map.entry("binding", "写入字段"),
            Map.entry("budget", "运行上限"),
            Map.entry("cameraAngle", "角度"),
            Map.entry("cameraEquipment", "设备"),
            Map.entry("cameraFixed", "固定镜头"),
            Map.entry("cameraMovement", "运镜"),
            Map.entry("canvasId", "画布编号"),
            Map.entry("canvasRunId", "画布运行编号"),
            Map.entry("character", "角色"),
            Map.entry("characterAssetIds", "关联角色"),
            Map.entry("characterIds", "角色变体"),
            Map.entry("character_asset_id", "角色资产"),
            Map.entry("character_asset_ids", "角色资产"),
            Map.entry("character_name", "角色名称"),
            Map.entry("characters", "角色名称"),
            Map.entry("charactersJson", "人物表"),
            Map.entry("clothing", "服装"),
            Map.entry("color", "颜色"),
            Map.entry("content", "正文"),
            Map.entry("contractId", "外部工具"),
            Map.entry("contractVersion", "MCP 操作版本"),
            Map.entry("count", "候选数量"),
            Map.entry("coverUrl", "封面"),
            Map.entry("createdCount", "创建数量"),
            Map.entry("customColumns", "自定义列"),
            Map.entry("customData", "扩展字段"),
            Map.entry("delete", "删除"),
            Map.entry("description", "描述"),
            Map.entry("detail", "细节"),
            Map.entry("detailLevel", "详情级别"),
            Map.entry("dialogue", "对白"),
            Map.entry("dialogues", "对白与动作"),
            Map.entry("duration", "时长（秒）"),
            Map.entry("durationEstimate", "预计时长（秒）"),
            Map.entry("environment", "环境"),
            Map.entry("episodeId", "分集"),
            Map.entry("episodeNumber", "集号"),
            Map.entry("episode_version", "分集版本"),
            Map.entry("expectedConfigVersion", "节点内容版本"),
            Map.entry("expectedLayoutVersion", "节点布局版本"),
            Map.entry("expectedParentVersion", "上级对象版本"),
            Map.entry("expectedRevision", "画布或任务版本"),
            Map.entry("expectedVersion", "预期版本"),
            Map.entry("expression", "表情"),
            Map.entry("ext", "外景"),
            Map.entry("false", "追加写入"),
            Map.entry("first", "首帧"),
            Map.entry("firstFrameImageUrl", "首帧"),
            Map.entry("firstFramePrompt", "首帧提示词"),
            Map.entry("fit", "适配方式"),
            Map.entry("focalLength", "焦距"),
            Map.entry("format", "导出格式"),
            Map.entry("fps", "帧率"),
            Map.entry("framePrompt", "画面提示词"),
            Map.entry("frameRate", "帧率"),
            Map.entry("frameType", "帧范围"),
            Map.entry("front", "正面"),
            Map.entry("full", "完整内容"),
            Map.entry("generateAudio", "生成音频"),
            Map.entry("generatedImageUrl", "生成图片"),
            Map.entry("generatedVideoUrl", "生成视频"),
            Map.entry("genre", "题材"),
            Map.entry("hairstyle", "发型"),
            Map.entry("height", "高度"),
            Map.entry("image", "图片模型"),
            Map.entry("imageModelId", "图片生成模型"),
            Map.entry("imageUrl", "画面图片"),
            Map.entry("imageUrls", "参考图片"),
            Map.entry("importance", "重要程度"),
            Map.entry("initial", "初始"),
            Map.entry("initialItem", "初始子资产"),
            Map.entry("input", "输入内容"),
            Map.entry("inputMode", "输入方式"),
            Map.entry("inputParameter", "接收内容的参数"),
            Map.entry("inputPort", "传入位置"),
            Map.entry("outputPort", "使用结果"),
            Map.entry("inputType", "输入类型"),
            Map.entry("instruction", "创作要求"),
            Map.entry("int", "内景"),
            Map.entry("intExt", "内外景"),
            Map.entry("int_ext", "内外景"),
            Map.entry("itemId", "子资产"),
            Map.entry("itemType", "变体类别"),
            Map.entry("items", "子项"),
            Map.entry("last", "尾帧"),
            Map.entry("lastFrameImageUrl", "尾帧"),
            Map.entry("lastFramePrompt", "尾帧提示词"),
            Map.entry("location", "地点"),
            Map.entry("manual", "手动创建"),
            Map.entry("material", "材质"),
            Map.entry("maxAgentRuns", "最多智能体任务"),
            Map.entry("maxBatchItems", "批量条目上限"),
            Map.entry("maxImages", "最多生成图片"),
            Map.entry("maxItems", "最大处理数量"),
            Map.entry("maxNodes", "最多执行次数"),
            Map.entry("maxParallel", "同时执行数量"),
            Map.entry("maxVideoSeconds", "最多视频秒数"),
            Map.entry("message", "任务说明"),
            Map.entry("modelId", "模型"),
            Map.entry("modelType", "模型类型"),
            Map.entry("music", "音乐"),
            Map.entry("musicVolume", "配乐音量"),
            Map.entry("name", "名称"),
            Map.entry("negativePrompt", "反向提示词"),
            Map.entry("nodeId", "节点编号"),
            Map.entry("nodeIds", "节点列表"),
            Map.entry("nodeRunId", "节点运行编号"),
            Map.entry("operation", "处理方式"),
            Map.entry("operationConfig", "处理设置"),
            Map.entry("operationKind", "批量处理能力"),
            Map.entry("operations", "画布修改"),
            Map.entry("original", "原始"),
            Map.entry("originalVolume", "原声音量"),
            Map.entry("outline", "大纲"),
            Map.entry("outputId", "结果编号"),
            Map.entry("overwrite", "覆盖"),
            Map.entry("overwriteMode", "写入方式"),
            Map.entry("parenthetical", "表演提示"),
            Map.entry("personality", "性格"),
            Map.entry("planHash", "运行计划校验值"),
            Map.entry("planId", "运行计划编号"),
            Map.entry("pose", "姿势"),
            Map.entry("projectId", "项目"),
            Map.entry("prompt", "提示词"),
            Map.entry("prop", "道具"),
            Map.entry("propAssetIds", "关联道具"),
            Map.entry("propIds", "道具变体"),
            Map.entry("prop_asset_ids", "道具资产"),
            Map.entry("properties", "项目扩展设置"),
            Map.entry("ratio", "画面比例"),
            Map.entry("rawContent", "原文"),
            Map.entry("referenceAudioUrls", "参考音频"),
            Map.entry("referenceImageUrl", "参考图片"),
            Map.entry("referenceImageUrls", "参考图片"),
            Map.entry("referenceVideoUrls", "参考视频"),
            Map.entry("remark", "备注"),
            Map.entry("replace", "替换"),
            Map.entry("resolution", "分辨率"),
            Map.entry("resourceId", "媒体素材"),
            Map.entry("resourceType", "资源类型"),
            Map.entry("resultType", "输出类型"),
            Map.entry("scene", "场景"),
            Map.entry("sceneAssetId", "关联场景"),
            Map.entry("sceneAssetItemId", "场景变体"),
            Map.entry("sceneCount", "场次数量"),
            Map.entry("sceneDescription", "场景描述"),
            Map.entry("sceneExpectation", "场景要求"),
            Map.entry("sceneHeading", "场景标题"),
            Map.entry("sceneNumber", "场号"),
            Map.entry("scene_asset_id", "场景资产"),
            Map.entry("scene_description", "场景描述"),
            Map.entry("scene_heading", "场次标题"),
            Map.entry("scenes", "场次"),
            Map.entry("scenes_only", "仅场次"),
            Map.entry("scriptEpisodeId", "对应剧本分集"),
            Map.entry("scriptEpisodeIds", "剧本分集"),
            Map.entry("scriptId", "剧本"),
            Map.entry("scriptSceneItemId", "剧本场次"),
            Map.entry("scriptSceneItemIds", "剧本场次"),
            Map.entry("seed", "随机种子"),
            Map.entry("selectedOutputId", "选定候选"),
            Map.entry("separator", "分隔文字"),
            Map.entry("sfxVolume", "音效音量"),
            Map.entry("shotCount", "镜头数量"),
            Map.entry("shotNumber", "镜号"),
            Map.entry("shotType", "景别"),
            Map.entry("shots", "镜头"),
            Map.entry("side", "侧面"),
            Map.entry("sortOrder", "排序"),
            Map.entry("sound", "声音"),
            Map.entry("soundEffect", "音效"),
            Map.entry("sourceType", "来源类型"),
            Map.entry("status", "项目状态"),
            Map.entry("storySynopsis", "故事梗概"),
            Map.entry("storyboardEpisodeId", "分镜分集"),
            Map.entry("storyboardId", "分镜"),
            Map.entry("storyboardItemId", "分镜条目"),
            Map.entry("storyboardSceneId", "分镜场次"),
            Map.entry("style", "画面风格"),
            Map.entry("summary", "摘要"),
            Map.entry("synopsis", "梗概"),
            Map.entry("tags", "标签"),
            Map.entry("targetAudience", "目标受众"),
            Map.entry("targetNodeId", "目标节点"),
            Map.entry("targetType", "目标类型"),
            Map.entry("thumbnailUrl", "缩略图"),
            Map.entry("timeOfDay", "时间"),
            Map.entry("time_of_day", "时间段"),
            Map.entry("title", "名称"),
            Map.entry("transition", "转场"),
            Map.entry("true", "覆盖已有数据"),
            Map.entry("type", "类型"),
            Map.entry("value", "内容"),
            Map.entry("valueType", "内容类型"),
            Map.entry("variant", "变体"),
            Map.entry("video", "视频模型"),
            Map.entry("videoModelId", "视频生成模型"),
            Map.entry("videoPrompt", "视频提示词"),
            Map.entry("videoUrl", "镜头视频"),
            Map.entry("watermark", "水印"),
            Map.entry("width", "宽度"),
            Map.entry("workflowParameters", "工作流设置"),
            Map.entry("workflowVersionId", "工作流版本"),
            Map.entry("x", "裁剪左边距"),
            Map.entry("y", "裁剪上边距"));
    private static final Map<String, String> CHOICES = Map.ofEntries(
            Map.entry("INHERIT", "跟随智能体"),
            Map.entry("CUSTOM", "单独指定"),
            Map.entry("JSON_OBJECT", "传入完整参数"),
            Map.entry("NONE", "使用固定输入"),
            Map.entry("RETRY_FAILED", "重试失败项"),
            Map.entry("STOP", "结束批次"),
            Map.entry("TEXT_FIELD", "传入文本"),
            Map.entry("USE_AVAILABLE", "采用已有结果"),
            Map.entry("ai_media", "综合创作助手"),
            Map.entry("asset", "资产"),
            Map.entry("assetItem", "子资产"),
            Map.entry("asset_image_executor", "单个资产生图"),
            Map.entry("asset_image_gen", "资产图片生成"),
            Map.entry("audio", "音频"),
            Map.entry("both", "首尾帧"),
            Map.entry("canvas_asset_extractor", "提取资产"),
            Map.entry("concept_visualizer", "概念与画面方案"),
            Map.entry("contain", "完整包含"),
            Map.entry("cover", "裁切填满"),
            Map.entry("crop", "裁剪"),
            Map.entry("csv", "CSV 表格"),
            Map.entry("document", "PDF 文档"),
            Map.entry("episode_scene_writer", "分集场次编写"),
            Map.entry("episode_script_creator", "单集创作"),
            Map.entry("episode_storyboard_writer", "单集分镜编写"),
            Map.entry("fill", "拉伸填满"),
            Map.entry("first", "首帧"),
            Map.entry("image", "图片"),
            Map.entry("json", "JSON 数据"),
            Map.entry("last", "尾帧"),
            Map.entry("markdown", "Markdown 文档"),
            Map.entry("media", "媒体文件"),
            Map.entry("music", "配乐"),
            Map.entry("project", "项目"),
            Map.entry("resize", "调整尺寸"),
            Map.entry("script", "剧本"),
            Map.entry("scriptEpisode", "剧本分集"),
            Map.entry("scriptScene", "剧本场次"),
            Map.entry("script_assistant", "剧本修改助手"),
            Map.entry("script_episode_parse", "分集解析"),
            Map.entry("script_full_parse", "整本剧本解析"),
            Map.entry("script_to_storyboard", "剧本转分镜"),
            Map.entry("sfx", "音效"),
            Map.entry("shot", "镜头"),
            Map.entry("story_to_script", "故事转剧本"),
            Map.entry("storyboard", "分镜"),
            Map.entry("storyboardEpisode", "分镜分集"),
            Map.entry("storyboardScene", "分镜场次"),
            Map.entry("storyboard_asset_preprocessor", "分镜资产准备"),
            Map.entry("storyboard_frame_executor", "单个镜头首尾帧"),
            Map.entry("storyboard_frame_gen", "镜头首尾帧生成"),
            Map.entry("storyboard_video_executor", "单个镜头视频"),
            Map.entry("storyboard_video_gen", "镜头视频生成"),
            Map.entry("text", "文本"),
            Map.entry("video", "视频"),
            Map.entry("zip", "ZIP 资源包"));

    public String describe(JsonNode value, CanvasActor actor) {
        return value("", value, value, actor);
    }

    public String describeValue(JsonNode value, CanvasValueType type, CanvasActor actor) {
        if (Set.of(CanvasValueType.IMAGE, CanvasValueType.VIDEO, CanvasValueType.AUDIO, CanvasValueType.DOCUMENT)
                .contains(type)
                && value.isTextual() && value.asText().startsWith("media://")) {
            return media.get(value.asText().substring(8).split("/")[0], actor).name();
        }
        return describe(value, actor);
    }

    public String describeResourceFields(JsonNode fields, CanvasValueType type, CanvasActor actor) {
        if (type == CanvasValueType.PROJECT && fields.isObject() && fields.path("status").isNumber()) {
            ObjectNode content = fields.deepCopy();
            content.put("status", switch (fields.path("status").asInt()) {
                case 0 -> "筹备";
                case 1 -> "制作中";
                case 2 -> "已完成";
                case 3 -> "已归档";
                default -> throw new BusinessException(400, "请选择有效的项目状态");
            });
            return describe(content, actor);
        }
        return describe(fields, actor);
    }

    public String resourceName(CanvasValueType type, long id, CanvasActor actor) {
        var data = resources.get(type, id, actor).data();
        for (String field : List.of("name", "title", "sceneHeading", "sceneNumber", "shotNumber")) {
            if (data.hasNonNull(field) && !data.path(field).asText().isBlank()) {
                return data.path(field).asText();
            }
        }
        return CHOICES.getOrDefault(type.code(), "项目内容");
    }

    private String value(String key, JsonNode value, JsonNode parent, CanvasActor actor) {
        if (value.isMissingNode() || value.isNull() || value.isTextual() && value.asText().isEmpty()) {
            return "（空）";
        }
        if (Set.of("properties", "customData", "customColumns", "workflowParameters", "argumentsJson").contains(key)) {
            return value.isTextual() ? value.asText() : value.toPrettyString();
        }
        if (value.isBoolean()) {
            return value.asBoolean() ? "开启" : "关闭";
        }
        if (value.isArray()) {
            var parts = new ArrayList<String>();
            for (var item : value) {
                parts.add(value(key, item, parent, actor));
            }
            return parts.isEmpty() ? "无" : String.join("\n\n", parts);
        }
        if (value.isObject()) {
            if (value.path("type").isInt() && value.path("content").isTextual()) {
                String speaker = value.path("character_name").asText("");
                String direction = value.path("parenthetical").asText("");
                return (speaker.isBlank() ? "" : speaker + (direction.isBlank() ? "" : "（" + direction + "）") + "：") + value.path(
                        "content").asText();
            }
            var parts = new ArrayList<String>();
            value.fields().forEachRemaining(entry -> {
                if (!INTERNAL.contains(entry.getKey())) {
                    parts.add(label(entry.getKey()) + "：" + value(entry.getKey(), entry.getValue(), value, actor));
                }
            });
            return parts.isEmpty() ? "（空）" : String.join("\n", parts);
        }
        if (Set.of("modelId", "imageModelId", "videoModelId").contains(key)) {
            return models.require(value.asLong()).name();
        }
        if (key.equals("apiConfigId")) {
            return models.audioServices().stream().filter(item -> item.id() == value.asLong()).findFirst()
                    .orElseThrow(() -> new BusinessException(400, "所选音频服务不可用，请重新选择")).name();
        }
        if (key.equals("resourceId")) {
            return media.get(value.asText(), actor).name();
        }
        if (key.equals("contractId")) {
            return contracts.require(value.asText(), actor.userId(), false).name();
        }
        if (key.equals("workflowVersionId")) {
            return models.list().stream().flatMap(model -> model.versions().stream())
                    .filter(item -> item.id() == value.asLong()).map(item -> "发布版本 " + item.number()).findFirst()
                    .orElseThrow(() -> new BusinessException(400, "所选工作流版本不可用，请重新选择"));
        }
        CanvasValueType resource = switch (key) {
            case "projectId" -> CanvasValueType.PROJECT;
            case "scriptId" -> CanvasValueType.SCRIPT;
            case "scriptEpisodeId" -> CanvasValueType.SCRIPT_EPISODE;
            case "storyboardId" -> CanvasValueType.STORYBOARD;
            case "storyboardEpisodeId" -> CanvasValueType.STORYBOARD_EPISODE;
            case "storyboardSceneId" -> CanvasValueType.STORYBOARD_SCENE;
            case "characterAssetIds", "propAssetIds", "sceneAssetId", "assetId", "character_asset_ids",
                 "prop_asset_ids", "scene_asset_id" -> CanvasValueType.ASSET;
            case "assetItemId", "sceneAssetItemId", "propAssetItemIds", "characterAssetItemIds" ->
                    CanvasValueType.ASSET_ITEM;
            default -> null;
        };
        if (resource != null) {
            return resourceName(resource, value.asLong(), actor);
        }
        if (key.equals("binding")) {
            return CanvasWritebackFields.list().stream().filter(field -> field.key().equals(value.asText()))
                    .map(field -> field.label()).findFirst()
                    .orElseThrow(() -> new BusinessException(400, "请选择可写入的内容字段"));
        }
        if (key.equals("status") && value.isNumber()) {
            return switch (value.asInt()) {
                case 0 -> "草稿";
                case 1 -> "正常";
                case 2 -> "已完成";
                case 3 -> "已归档";
                default -> "状态已更新";
            };
        }
        String text = value.asText();
        if (key.equals("type") || key.equals("itemType")) {
            return switch (text) {
                case "character" -> "角色";
                case "scene" -> "场景";
                case "prop" -> "道具";
                case "variant" -> "变体";
                case "outfit" -> "服装";
                case "expression" -> "表情";
                case "pose" -> "姿势";
                case "angle" -> "角度";
                default -> choiceLabel(text);
            };
        }
        if (key.equals("operationKind")) {
            return capabilities.require(CanvasNodeKind.fromCode(text)).label();
        }
        if (key.equals("inputPort") || key.equals("outputPort")) {
            return capabilities.list().stream()
                    .flatMap(node -> (key.equals("inputPort") ? node.inputs() : node.outputs()).stream())
                    .filter(port -> port.id().equals(text)).map(port -> port.label()).findFirst()
                    .orElseThrow(() -> new BusinessException(400, "请重新选择输入或输出位置"));
        }
        if ((key.endsWith("Url") || key.endsWith("Urls")) && text.startsWith("media://")) {
            return media.get(text.substring(8).split("/")[0], actor).name();
        }
        if (Set.of("valueType", "targetType", "inputType", "resultType", "operationKind", "agentType", "frameType",
                "fit", "operation", "inputMode", "format", "skillMode").contains(key)) {
            return choiceLabel(text);
        }
        return text;
    }

    String choiceLabel(String value) {
        String known = CHOICES.get(value);
        if (known != null) {
            return known;
        }
        return capabilities.list().stream().filter(node -> node.kind().code().equals(value)).map(node -> node.label())
                .findFirst()
                .orElseThrow(() -> new BusinessException(400, "助手暂时无法说明这项设置，请在节点编辑器中重新选择。"));
    }

    static String label(String key) {
        String label = LABELS.get(key);
        if (label == null) {
            throw new BusinessException(400, "助手暂时无法说明这项修改，请改用节点编辑器完成。");
        }
        return label;
    }
}
