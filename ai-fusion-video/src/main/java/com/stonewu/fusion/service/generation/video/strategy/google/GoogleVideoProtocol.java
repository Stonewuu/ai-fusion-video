package com.stonewu.fusion.service.generation.video.strategy.google;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.ai.google.GoogleRequestErrors;
import com.stonewu.fusion.service.ai.google.GoogleInteractionsProtocol;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Omni 使用 Interactions，Veo 使用长任务；保留各自的输入和响应结构。 */
public final class GoogleVideoProtocol {
    private GoogleVideoProtocol() {
    }

    public static boolean omni(AiModel model) {
        return model.getCode().startsWith("gemini-omni-");
    }

    public static JSONObject request(VideoTask task, AiModel model, JSONObject config, boolean developer) {
        validate(task, model, config);
        if (Boolean.FALSE.equals(task.getGenerateAudio()) && (omni(model) || developer && !model.getCode().startsWith("veo-2"))) {
            throw new BusinessException("此模型随视频生成声音，不能单独关闭");
        }
        return omni(model) ? omniRequest(task, model, config, developer) : veoRequest(task, model, config, developer);
    }

    public static void validate(VideoTask task, AiModel model, JSONObject config) {
        if (!omni(model) && !model.getCode().startsWith("veo-")) {
            throw new BusinessException("请选择 Omni 或 Veo 视频模型");
        }
        int duration = task.getDuration() == null ? config.getInt("defaultDuration", 8) : task.getDuration();
        int min = config.getInt("minDuration", omni(model) ? 3 : 4);
        int max = config.getInt("maxDuration", omni(model) ? model.getCode().startsWith("gemini-omni-1.2-") ? 15 : 10 : 8);
        if (duration < min || duration > max) {
            throw new BusinessException("视频时长应为 " + min + "–" + max + " 秒");
        }
        if (omni(model) && StrUtil.isNotBlank(task.getLastFrameImageUrl()) && duration > 10) {
            throw new BusinessException("使用首尾帧时，视频时长不能超过 10 秒");
        }
        if (!omni(model) && !model.getCode().startsWith("veo-2") && !List.of(4, 6, 8).contains(duration)) {
            throw new BusinessException("Veo 视频时长请选择 4、6 或 8 秒");
        }
        String resolution = task.getResolution() == null ? config.getStr("defaultResolution", "720p")
                : task.getResolution().toLowerCase(Locale.ROOT);
        var resolutions = config.getJSONArray("supportedResolutions");
        if (resolutions != null && resolutions.toList(String.class).stream().noneMatch(resolution::equalsIgnoreCase)) {
            throw new BusinessException("请选择模型支持的分辨率");
        }
        var ratios = config.getJSONArray("supportedAspectRatios");
        if (task.getRatio() != null && ratios != null && !ratios.contains(task.getRatio())) {
            throw new BusinessException("请选择模型支持的画面比例");
        }
        if (!omni(model) && (resolution.equalsIgnoreCase("1080p") || resolution.equalsIgnoreCase("4k")
                || !list(task.getReferenceImageUrls()).isEmpty()) && duration != 8) {
            throw new BusinessException("Veo 使用高清分辨率或参考图时，视频时长请选择 8 秒");
        }
        if (!omni(model) && !list(task.getReferenceVideoUrls()).isEmpty()
                && (duration != 8 || !resolution.equalsIgnoreCase("720p"))) {
            throw new BusinessException("Veo 延长视频请选择 720p、8 秒");
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl()) && StrUtil.isBlank(task.getFirstFrameImageUrl())) {
            throw new BusinessException("请先添加视频首帧");
        }
        if (!list(task.getReferenceAudioUrls()).isEmpty()) {
            throw new BusinessException("此视频模型不支持参考音频，请移除后重试");
        }
        if (omni(model) && duration > 10 && list(task.getReferenceVideoUrls()).size() > 1) {
            throw new BusinessException("超过 10 秒的视频最多使用 1 个参考视频");
        }
        if (task.getSeed() != null && (task.getSeed() < 0 || task.getSeed() > 4294967295L)) {
            throw new BusinessException("随机种子应为 0–4294967295");
        }
    }

    private static JSONObject omniRequest(VideoTask task, AiModel model, JSONObject config, boolean developer) {
        var input = new JSONArray();
        addFrame(input, task.getFirstFrameImageUrl(), "Use the following image as the first frame.");
        addFrame(input, task.getLastFrameImageUrl(), "Use the following image as the last frame.");
        for (String image : list(task.getReferenceImageUrls())) {
            input.add(interactionMedia(image, "image"));
        }
        for (String video : list(task.getReferenceVideoUrls())) {
            input.add(interactionMedia(video, "video"));
        }
        input.add(new JSONObject().set("type", "text").set("text", task.getPrompt()));
        var format = new JSONObject().set("type", "video").set("delivery", "inline");
        if (task.getRatio() != null) {
            format.set("aspect_ratio", task.getRatio());
        }
        if (task.getResolution() != null) {
            format.set("resolution", task.getResolution().toLowerCase(Locale.ROOT));
        }
        format.set("duration", (task.getDuration() == null ? config.getInt("defaultDuration", 8) : task.getDuration()) + "s");
        var request = new JSONObject().set("model", model.getCode()).set("input", GoogleInteractionsProtocol.input(input))
                .set("background", true).set("store", true).set("stream", false)
                .set("response_format", GoogleInteractionsProtocol.format(format, developer));
        if (task.getSeed() != null) {
            request.set("generation_config", new JSONObject().set("seed", task.getSeed()));
        }
        return request;
    }

    private static void addFrame(JSONArray input, String source, String instruction) {
        if (StrUtil.isNotBlank(source)) {
            input.add(new JSONObject().set("type", "text").set("text", instruction));
            input.add(interactionMedia(source, "image"));
        }
    }

    public static JSONObject interactionMedia(String source, String kind) {
        var value = new JSONObject().set("type", kind);
        if (!source.startsWith("data:")) {
            return value.set("uri", source);
        }
        var inline = inlineData(source);
        if (!inline.getStr("mimeType").startsWith(kind + "/")) {
            throw new BusinessException("参考素材的类型不符，请重新选择");
        }
        return value.set("data", inline.getStr("data")).set("mime_type", inline.getStr("mimeType"));
    }

    public static JSONObject inlineData(String source) {
        int separator = source.indexOf(";base64,");
        if (!source.startsWith("data:") || separator < 5 || separator + 8 >= source.length()) {
            throw new BusinessException("参考素材内容无效，请重新上传");
        }
        return new JSONObject().set("mimeType", source.substring(5, separator))
                .set("data", source.substring(separator + 8));
    }

    private static JSONObject veoMedia(String source, boolean developer) {
        var inline = inlineData(source);
        return developer ? new JSONObject().set("inlineData", inline)
                : new JSONObject().set("bytesBase64Encoded", inline.getStr("data"))
                .set("mimeType", inline.getStr("mimeType"));
    }

    private static JSONObject veoRequest(VideoTask task, AiModel model, JSONObject config, boolean developer) {
        var instance = new JSONObject().set("prompt", task.getPrompt());
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            instance.set("image", veoMedia(task.getFirstFrameImageUrl(), developer));
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            instance.set("lastFrame", veoMedia(task.getLastFrameImageUrl(), developer));
        }
        var images = new JSONArray();
        for (String image : list(task.getReferenceImageUrls())) {
            images.add(new JSONObject().set("image", veoMedia(image, developer)).set("referenceType", "asset"));
        }
        if (!images.isEmpty()) {
            instance.set("referenceImages", images);
        }
        var videos = list(task.getReferenceVideoUrls());
        if (!videos.isEmpty()) {
            if (videos.size() != 1) {
                throw new BusinessException("Veo 每次只能延长 1 个视频");
            }
            instance.set("video", veoMedia(videos.getFirst(), developer));
        }
        var parameters = new JSONObject().set("sampleCount", 1)
                .set("durationSeconds", task.getDuration() == null ? config.getInt("defaultDuration", 8) : task.getDuration());
        if (task.getRatio() != null) {
            parameters.set("aspectRatio", task.getRatio());
        }
        if (task.getResolution() != null && !model.getCode().startsWith("veo-2")) {
            parameters.set("resolution", task.getResolution().toLowerCase(Locale.ROOT));
        }
        if (task.getSeed() != null) {
            parameters.set("seed", task.getSeed());
        }
        if (!developer && !model.getCode().startsWith("veo-2")) {
            parameters.set("generateAudio", task.getGenerateAudio() == null || task.getGenerateAudio());
        }
        if (config.containsKey("personGeneration")) {
            parameters.set("personGeneration", config.getStr("personGeneration"));
        }
        return new JSONObject().set("instances", new JSONArray().put(instance)).set("parameters", parameters);
    }

    public static RemoteGenerationResult result(JSONObject response, boolean omni, boolean developer, Integer duration) {
        if (response.containsKey("error")) {
            return RemoteGenerationResult.failed(GoogleRequestErrors.operationMessage(response.getJSONObject("error")));
        }
        if (omni) {
            String status = response.getStr("status", "");
            if (List.of("in_progress", "queued").contains(status)) {
                return RemoteGenerationResult.pending(5);
            }
            if ("cancelled".equals(status)) {
                return RemoteGenerationResult.cancelled("视频生成已取消");
            }
            if (!"completed".equals(status)) {
                return RemoteGenerationResult.failed("视频生成未完成，请调整提示词后重试");
            }
            var outputs = new ArrayList<RemoteGenerationResult.Output>();
            JSONArray steps = response.getJSONArray("steps");
            if (steps != null) {
                for (JSONObject step : steps.toList(JSONObject.class)) {
                    if (!"model_output".equals(step.getStr("type")) || step.getJSONArray("content") == null) {
                        continue;
                    }
                    for (JSONObject content : step.getJSONArray("content").toList(JSONObject.class)) {
                        if ("video".equals(content.getStr("type"))) {
                            outputs.add(RemoteGenerationResult.Output.video(media(content, "data", "mime_type", "uri"), null, duration));
                        }
                    }
                }
            }
            return RemoteGenerationResult.success(outputs);
        }
        if (!response.getBool("done", false)) {
            return RemoteGenerationResult.pending(10);
        }
        var payload = response.getJSONObject("response");
        var videos = payload == null ? null : developer
                ? payload.getByPath("generateVideoResponse.generatedSamples", JSONArray.class) : payload.getJSONArray("videos");
        var outputs = new ArrayList<RemoteGenerationResult.Output>();
        if (videos != null) {
            for (JSONObject item : videos.toList(JSONObject.class)) {
                JSONObject video = developer ? item.getJSONObject("video") : item;
                outputs.add(RemoteGenerationResult.Output.video(media(video,
                        developer ? "data" : "bytesBase64Encoded", "mimeType", developer ? "uri" : "gcsUri"), null, duration));
            }
        }
        return RemoteGenerationResult.success(outputs);
    }

    private static String media(JSONObject media, String dataKey, String mimeKey, String uriKey) {
        if (media == null) {
            throw new BusinessException("Google 未返回视频内容");
        }
        String data = media.getStr(dataKey);
        if (StrUtil.isNotBlank(data)) {
            return "data:" + media.getStr(mimeKey, "video/mp4") + ";base64," + data;
        }
        String uri = media.getStr(uriKey);
        if (StrUtil.isBlank(uri)) {
            throw new BusinessException("Google 未返回视频内容");
        }
        return uri;
    }

    private static List<String> list(String json) {
        return StrUtil.isBlank(json) ? List.of() : JSONUtil.parseArray(json).toList(String.class);
    }
}
