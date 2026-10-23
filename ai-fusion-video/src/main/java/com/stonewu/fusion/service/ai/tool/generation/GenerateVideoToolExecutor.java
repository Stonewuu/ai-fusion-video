package com.stonewu.fusion.service.ai.tool.generation;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeGeneration;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.resource.ResourceToolRequests;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 提交可恢复生成任务；返回真实任务引用，不阻塞工具线程。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GenerateVideoToolExecutor implements ToolExecutor {

    /**
     * 模型类型常量：视频生成
     */
    private static final int MODEL_TYPE_VIDEO = 3;


    private final AiModelService aiModelService;
    private final VideoGenerationService videoGenerationService;
    private final GenerationSubmissionService submissions;
    private final GenerationModelCapabilityService generationModelCapabilityService;
    private final CanvasJson json;
    private final CanvasAgentNodeGeneration canvasModels;

    @Override
    public String getToolName() {
        return "generate_video";
    }

    @Override
    public String getDisplayName() {
        return "AI 生成视频";
    }

    @Override
    public String getToolDescription() {
        return """
                生成AI视频。根据提示词和可选的参考图片生成视频片段。

                适用场景：
                1. 为分镜镜头生成视频：根据画面描述、运镜指令和参考首帧图生成视频
                2. 文本生成视频：纯文字描述生成视频
                3. 多模态参考生视频：传入角色/道具/场景的参考图片、参考视频、参考音频，提升画面中人物和物品的一致性

                重要提示：
                - 提示词应详细描述画面内容、运镜方式、环境氛围
                - 提供 firstFrameImageUrl 可大幅提升画面一致性
                - 提供 referenceImageUrls 可锁定角色形象、道具外观等，须符合所选模型能力
                - 提供 referenceVideoUrls 可参考视频的动作、运镜、特效等
                - 提供 referenceAudioUrls 可参考音色、音乐旋律、对话内容等
                - 使用多模态参考时，提示词中用`图片1`/`视频1`/`音频1`等指代对应的参考素材
                - 生成耗时较长（通常 1-5 分钟），请耐心等待
                - 如果你打算传首帧、尾帧、参考图、参考视频或参考音频，或不确定当前默认模型是否支持这些字段，请先调用 get_generation_model_capabilities
                
                画布专业节点使用已明确选择的模型；其他对话使用配置的默认模型。实际限制请查询能力工具。
                """;
    }

    @Override
    public String getParametersSchema() {
        String firstFrameDescription = "首帧参考图片 URL，调用前查询选定模型能力";
        String lastFrameDescription = "尾帧参考图片 URL，调用前查询选定模型能力";
        String referenceImageDescription = "参考图片 URL 列表，须符合选定模型能力";
        String referenceVideoDescription = "参考视频 URL 列表，须符合选定模型能力";
        String referenceAudioDescription = "参考音频 URL 列表，须符合选定模型能力";

        return JSONUtil.createObj()
                .set("type", "object")
                .set("properties", JSONUtil.createObj()
                        .set("prompt", JSONUtil.createObj()
                                .set("type", "string")
                                .set("description",
                                        "视频生成提示词；是否允许仅使用素材由模型 videoRules.promptRequired 决定"))
                        .set("firstFrameImageUrl", JSONUtil.createObj()
                                .set("type", "string")
                                .set("description", firstFrameDescription))
                        .set("lastFrameImageUrl", JSONUtil.createObj()
                                .set("type", "string")
                                .set("description", lastFrameDescription))
                        .set("referenceImageUrls", JSONUtil.createObj()
                                .set("type", "array")
                                .set("items", JSONUtil.createObj().set("type", "string"))
                                .set("description", referenceImageDescription))
                        .set("referenceVideoUrls", JSONUtil.createObj()
                                .set("type", "array")
                                .set("items", JSONUtil.createObj().set("type", "string"))
                                .set("description", referenceVideoDescription))
                        .set("referenceAudioUrls", JSONUtil.createObj()
                                .set("type", "array")
                                .set("items", JSONUtil.createObj().set("type", "string"))
                                .set("description", referenceAudioDescription))
                        .set("ratio", JSONUtil.createObj()
                                .set("type", "string")
                                .set("description", "画面比例，省略时使用所选模型默认值；首尾帧可能必须使用 adaptive"))
                        .set("resolution", JSONUtil.createObj().set("type", "string")
                                .set("description", "输出分辨率，须使用模型能力返回的枚举值"))
                        .set("duration", JSONUtil.createObj()
                                .set("type", "integer")
                                .set("description",
                                        "视频时长（秒），省略时使用模型默认值；仅 automaticDuration=true 的模型允许 -1 自动时长，额度按 maxDuration 计算"))
                        .set("generateAudio", JSONUtil.createObj().set("type", "boolean")
                                .set("description", "是否生成声音，仅支持声音开关的模型可设；原生有声模型须省略"))
                        .set("watermark", JSONUtil.createObj().set("type", "boolean")
                                .set("description", "是否加水印，仅支持此开关的模型可设"))
                        .set("seed", JSONUtil.createObj().set("type", "integer")
                                .set("description", "随机种子，仅支持种子的模型可设"))
                        .set("cameraFixed", JSONUtil.createObj()
                                .set("type", "boolean")
                                .set("description", "是否固定镜头（不做运动），默认 false")))
                .toString();
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            String prompt = params.getStr("prompt");

            String firstFrameImageUrl = params.getStr("firstFrameImageUrl");
            String lastFrameImageUrl = params.getStr("lastFrameImageUrl");
            String ratio = params.getStr("ratio");
            Integer duration = params.getInt("duration");
            Boolean cameraFixed = params.getBool("cameraFixed");

            // 解析多模态参考图片列表
            List<String> referenceImageUrlList = new ArrayList<>();
            JSONArray refImagesArr = params.getJSONArray("referenceImageUrls");
            if (refImagesArr != null) {
                for (int i = 0; i < refImagesArr.size(); i++) {
                    String url = refImagesArr.getStr(i);
                    if (StrUtil.isNotBlank(url)) {
                        referenceImageUrlList.add(url);
                    }
                }
            }
            String referenceImageUrls = CollUtil.isEmpty(referenceImageUrlList)
                    ? null : JSONUtil.toJsonStr(referenceImageUrlList);

            // 解析参考视频列表
            String referenceVideoUrls = parseUrlArray(params, "referenceVideoUrls");

            // 解析参考音频列表
            String referenceAudioUrls = parseUrlArray(params, "referenceAudioUrls");

            // 确定生成模式
            String generateMode = StrUtil.isNotBlank(firstFrameImageUrl) ? "image2video" : "text2video";

            AiModel model = canvasModels.scoped(context) ? canvasModels.require(context,
                    GenerationKind.VIDEO) : resolvePreferredModel();
            var config = generationModelCapabilityService.getMergedModelConfig(model);
            // 旧模型保留原工具默认值；新契约由统一校验应用模型默认值。
            if (!config.containsKey("videoRules")) {
                if (StrUtil.isBlank(prompt)) {
                    return errorResult("请填写创作要求");
                }
                if (ratio == null) {
                    ratio = "16:9";
                }
                if (duration == null) {
                    duration = 5;
                }
            }

            // 构建生视频任务
            VideoTask task = VideoTask.builder()
                    .prompt(prompt)
                    .generateMode(generateMode)
                    .firstFrameImageUrl(firstFrameImageUrl)
                    .lastFrameImageUrl(lastFrameImageUrl)
                    .referenceImageUrls(referenceImageUrls)
                    .referenceVideoUrls(referenceVideoUrls)
                    .referenceAudioUrls(referenceAudioUrls)
                    .ratio(ratio)
                    .resolution(params.getStr("resolution"))
                    .duration(duration)
                    .generateAudio(params.getBool("generateAudio"))
                    .watermark(params.getBool("watermark"))
                    .seed(params.getLong("seed"))
                    .cameraFixed(cameraFixed)
                    .modelId(model.getId())
                    .count(1)
                    .userId(context.getUserId())
                    .build();

            generationModelCapabilityService.validateVideoTask(model, task);

            log.info(
                    "[generate_video] 提交生视频任务: prompt={}, mode={}, ratio={}, duration={}s, modelId={}, modelCode={}, 首帧: {}, 参考图: {}张",
                    StrUtil.sub(prompt, 0, 80), generateMode, ratio, duration, model.getId(), model.getCode(),
                    firstFrameImageUrl != null ? "有" : "无",
                    referenceImageUrlList.size());

            var mutation = context.requireMutationContext();
            var accepted = submissions.video(task, ResourceToolRequests.requestId(json, mutation), mutation, null);
            return JSONUtil.createObj()
                    .set("status", "pending")
                    .set("submissionId", accepted.id())
                    .set("taskId", accepted.publicTaskId())
                    .set("recordId", accepted.taskId())
                    .set("state", accepted.state())
                    .set("message",
                            "生成已可靠受理。通过 get_generation_task 读取实际结果；尚未完成时不要将其报告为生成成功。")
                    .toString();

        } catch (Exception e) {
            log.error("[generate_video] 生成视频失败", e);
            return errorResult("生成失败: " + e.getMessage());
        }
    }

    /**
     * 获取默认视频生成模型的 ID
     */
    private AiModel resolvePreferredModel() {
        AiModel defaultModel = aiModelService.getDefaultByType(MODEL_TYPE_VIDEO);
        if (defaultModel != null) {
            return defaultModel;
        }
        throw new IllegalStateException("未配置可用的视频生成模型");
    }

    private String errorResult(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }

    /**
     * 从参数中解析 URL 数组字段，返回 JSON 字符串或 null
     */
    private String parseUrlArray(JSONObject params, String fieldName) {
        JSONArray arr = params.getJSONArray(fieldName);
        if (arr == null || arr.isEmpty()) {
            return null;
        }
        List<String> urls = new ArrayList<>();
        for (int i = 0; i < arr.size(); i++) {
            String url = arr.getStr(i);
            if (StrUtil.isNotBlank(url)) {
                urls.add(url);
            }
        }
        return CollUtil.isEmpty(urls) ? null : JSONUtil.toJsonStr(urls);
    }
}
