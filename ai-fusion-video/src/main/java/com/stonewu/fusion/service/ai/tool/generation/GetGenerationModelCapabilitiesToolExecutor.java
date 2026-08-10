package com.stonewu.fusion.service.ai.tool.generation;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 查询当前默认生图/生视频模型能力。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GetGenerationModelCapabilitiesToolExecutor implements ToolExecutor {

    private static final int MODEL_TYPE_IMAGE = 2;
    private static final int MODEL_TYPE_VIDEO = 3;

    private final AiModelService aiModelService;
    private final GenerationModelCapabilityService generationModelCapabilityService;

    @Override
    public String getToolName() {
        return "get_generation_model_capabilities";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询生成模型能力";
    }

    @Override
    public String getToolDescription() {
        return """
                查询当前默认生图模型和生视频模型的能力约束。

                使用时机：
                - 在调用 generate_image 前，先确认当前默认图片模型是否支持 imageUrls
                - 在调用 generate_video 前，先确认当前默认视频模型是否支持 firstFrameImageUrl、lastFrameImageUrl、referenceImageUrls、referenceVideoUrls、referenceAudioUrls
                - 当你不确定默认模型能否使用参考图、首尾帧、多模态参考时，优先调用本工具，避免无意义重试

                默认返回图片和视频两类能力；也可以只查询 image 或 video。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "modelType": {
                            "type": "string",
                            "description": "查询类型：image、video、all（默认 all）",
                            "enum": ["image", "video", "all"]
                        }
                    },
                    "required": []
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = StrUtil.isBlank(toolInput) ? new JSONObject() : JSONUtil.parseObj(toolInput);
            String requestedType = StrUtil.blankToDefault(params.getStr("modelType"), "all").trim().toLowerCase();
            if (!"image".equals(requestedType) && !"video".equals(requestedType) && !"all".equals(requestedType)) {
                return errorResult("modelType 仅支持 image、video、all");
            }

            JSONObject result = JSONUtil.createObj()
                    .set("status", "success")
                    .set("requestedModelType", requestedType)
                    .set("usageHint", "先根据这里的能力结果组织 generate_image 或 generate_video 参数，只传当前默认模型支持的字段。")
                    .set("retryPolicy", "如果某个字段不受支持，请直接删除该字段并改写 prompt，不要重复用相同的不支持参数重试。\n");

            if (!"video".equals(requestedType)) {
                ResolvedModel imageModel = resolvePreferredModel(MODEL_TYPE_IMAGE);
                result.set("image", buildImageResult(imageModel));
            }
            if (!"image".equals(requestedType)) {
                ResolvedModel videoModel = resolvePreferredModel(MODEL_TYPE_VIDEO);
                result.set("video", buildVideoResult(videoModel));
            }
            return result.toString();
        } catch (Exception e) {
            log.error("查询生成模型能力失败", e);
            return errorResult("查询失败: " + e.getMessage());
        }
    }

    private JSONObject buildImageResult(ResolvedModel resolvedModel) {
        if (resolvedModel == null || resolvedModel.model() == null) {
            return JSONUtil.createObj()
                    .set("configured", false)
                    .set("selectionSource", null)
                    .set("toolGuidance", "当前没有可用的默认图片模型，请先配置图片模型。")
                    .set("summary", "当前未配置默认图片模型。");
        }

        JSONObject snapshot = generationModelCapabilityService.buildImageCapabilitySnapshot(resolvedModel.model());
        boolean supportsReferenceImages = snapshot.getBool("supportsReferenceImages", false);
        snapshot.set("selectionSource", resolvedModel.selectionSource());
        snapshot.set("toolGuidance", supportsReferenceImages
                ? "调用 generate_image 时可以传 imageUrls；如果传多张参考图，请在 prompt 中按顺序使用 图片1、图片2 等引用。"
                : "调用 generate_image 时不要传 imageUrls；把参考图中的风格和主体特征改写到 prompt 里。"
        );
        return snapshot;
    }

    private JSONObject buildVideoResult(ResolvedModel resolvedModel) {
        if (resolvedModel == null || resolvedModel.model() == null) {
            return JSONUtil.createObj()
                    .set("configured", false)
                    .set("selectionSource", null)
                    .set("toolGuidance", "当前没有可用的默认视频模型，请先配置视频模型。")
                    .set("summary", "当前未配置默认视频模型。");
        }

        JSONObject snapshot = generationModelCapabilityService.buildVideoCapabilitySnapshot(resolvedModel.model());
        boolean supportsFirstFrame = snapshot.getBool("supportsFirstFrame", false);
        boolean supportsReferenceImages = snapshot.getBool("supportsReferenceImages", false);
        boolean supportsReferenceVideos = snapshot.getBool("supportsReferenceVideos", false);
        boolean supportsReferenceAudios = snapshot.getBool("supportsReferenceAudios", false);

        snapshot.set("selectionSource", resolvedModel.selectionSource());
        snapshot.set("toolGuidance", buildVideoGuidance(
                supportsFirstFrame,
                supportsReferenceImages,
                supportsReferenceVideos,
                supportsReferenceAudios));
        return snapshot;
    }

    private String buildVideoGuidance(boolean supportsFirstFrame,
                                      boolean supportsReferenceImages,
                                      boolean supportsReferenceVideos,
                                      boolean supportsReferenceAudios) {
        List<String> hints = new ArrayList<>();
        hints.add(supportsFirstFrame
                ? "可以传 firstFrameImageUrl 锁定开场画面。"
                : "不要传 firstFrameImageUrl，改为在 prompt 中完整描述静态画面。"
        );
        hints.add(supportsReferenceImages
                ? "可以传 referenceImageUrls，用 图片1、图片2 等在 prompt 中引用。"
                : "不要传 referenceImageUrls，改为在 prompt 中文字描述角色、场景和道具特征。"
        );
        if (!supportsReferenceVideos) {
            hints.add("不要传 referenceVideoUrls。");
        }
        if (!supportsReferenceAudios) {
            hints.add("不要传 referenceAudioUrls。");
        }
        return String.join(" ", hints);
    }

    private ResolvedModel resolvePreferredModel(int modelType) {
        AiModel defaultModel = aiModelService.getDefaultByType(modelType);
        if (defaultModel != null) {
            return new ResolvedModel(defaultModel, "default_model");
        }

        List<AiModel> models = aiModelService.getListByType(modelType);
        if (!models.isEmpty()) {
            return new ResolvedModel(models.get(0), "first_enabled_fallback");
        }
        return null;
    }

    private String errorResult(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }

    private record ResolvedModel(AiModel model, String selectionSource) {
    }
}
