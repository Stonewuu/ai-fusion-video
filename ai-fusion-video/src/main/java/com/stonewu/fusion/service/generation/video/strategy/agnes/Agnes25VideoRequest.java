package com.stonewu.fusion.service.generation.video.strategy.agnes;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import okhttp3.MediaType;
import okhttp3.RequestBody;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Agnes Video 2.5 的原生 JSON 请求；与 2.0 的帧数协议分开。
 */
final class Agnes25VideoRequest {
    private static final Set<String> RATIOS = Set.of("21:9", "16:9", "4:3", "1:1", "3:4", "9:16");
    private static final Set<String> SIZES = Set.of("720P", "1080P", "1K", "2K");

    private Agnes25VideoRequest() {
    }

    static RequestBody build(OpenAiCompatibleVideoProtocolContext context) {
        VideoTask task = context.task();
        JSONObject config = context.modelConfig();
        String model = StrUtil.trim(context.model().getCode());
        String prompt = StrUtil.trim(task.getPrompt());
        if (StrUtil.isBlank(model) || StrUtil.isBlank(prompt)) {
            throw new BusinessException("请填写视频模型和生成提示词");
        }
        boolean flash = "agnes-video-2.5-flash".equals(model) || config.getBool("agnesVideoFlash", false);
        int seconds = task.getDuration() == null ? config.getInt("defaultDuration", 5) : task.getDuration();
        if (seconds < 4 || seconds > 12) {
            throw new BusinessException("Agnes 2.5 视频时长需为 4–12 秒");
        }
        String size = StrUtil.blankToDefault(task.getResolution(), config.getStr("defaultResolution", "720P"))
                .trim().toUpperCase(Locale.ROOT);
        if (!SIZES.contains(size) || flash && !"720P".equals(size)) {
            throw new BusinessException(flash ? "Agnes 2.5 Flash 仅支持 720P" : "请选择 Agnes 2.5 支持的视频分辨率");
        }
        String ratio = StrUtil.blankToDefault(task.getRatio(), config.getStr("defaultAspectRatio", "16:9")).trim();
        if (!RATIOS.contains(ratio)) {
            throw new BusinessException("请选择 Agnes 2.5 支持的画面比例");
        }
        if (Boolean.TRUE.equals(task.getWatermark()) || Boolean.TRUE.equals(task.getCameraFixed())
                || task.getGenerateAudio() != null) {
            throw new BusinessException("Agnes 2.5 不支持单独设置水印、固定镜头或声音开关，请在提示词中描述画面与声音");
        }

        String first = mediaUrl(task.getFirstFrameImageUrl(), "首帧图片");
        String last = mediaUrl(task.getLastFrameImageUrl(), "尾帧图片");
        List<String> images = mediaUrls(task.getReferenceImageUrls(), "参考图片");
        List<String> audios = mediaUrls(task.getReferenceAudioUrls(), "参考音频");
        List<String> videos = mediaUrls(task.getReferenceVideoUrls(), "参考视频");
        boolean frames = first != null || last != null;
        boolean references = !images.isEmpty() || !audios.isEmpty() || !videos.isEmpty();
        if (frames && references) {
            throw new BusinessException("首尾帧与参考素材不能同时使用，请保留一种输入方式");
        }
        if (images.size() > (flash ? 5 : 8)) {
            throw new BusinessException("参考图片超过当前 Agnes 视频模型的数量限制");
        }
        if (audios.size() > 3) {
            throw new BusinessException("Agnes 2.5 最多接收 3 段参考音频");
        }
        if (videos.size() > (flash ? 0 : 1)) {
            throw new BusinessException(flash ? "Agnes 2.5 Flash 不支持参考视频" : "Agnes 2.5 最多接收 1 段参考视频");
        }

        JSONObject body = JSONUtil.createObj().set("model", model).set("prompt", prompt)
                .set("seconds", String.valueOf(seconds)).set("size", size).set("aspect_ratio", ratio)
                .set("mode", frames ? "keyframe" : references ? "reference" : "text").set("n", 1);
        // 生成任务的多个结果由执行账本分别提交；每个远端请求只生成一个视频。
        if (task.getSeed() != null) {
            body.set("seed", task.getSeed());
        }
        if (first != null) {
            body.set("first_frame", first);
        }
        if (last != null) {
            body.set("last_frame", last);
        }
        if (!images.isEmpty()) {
            body.set("images", images);
        }
        if (!audios.isEmpty()) {
            body.set("audios", audios);
        }
        if (!videos.isEmpty()) {
            body.set("videos", videos.stream().map(url -> JSONUtil.createObj().set("url", url)).toList());
        }
        return RequestBody.create(body.toString(), MediaType.get("application/json"));
    }

    private static List<String> mediaUrls(String json, String label) {
        if (StrUtil.isBlank(json)) {
            return List.of();
        }
        JSONArray values;
        try {
            values = JSONUtil.parseArray(json);
        } catch (RuntimeException failure) {
            throw new BusinessException(label + "列表格式不正确");
        }
        List<String> urls = new ArrayList<>();
        for (Object value : values) {
            if (!(value instanceof String text) || StrUtil.isBlank(text)) {
                throw new BusinessException(label + "列表包含无效链接");
            }
            urls.add(mediaUrl(text, label));
        }
        return urls;
    }

    private static String mediaUrl(String value, String label) {
        if (StrUtil.isBlank(value)) {
            return null;
        }
        String url = value.trim();
        try {
            URI uri = URI.create(url);
            if (!("https".equalsIgnoreCase(uri.getScheme()) || "http".equalsIgnoreCase(uri.getScheme()))
                    || StrUtil.isBlank(uri.getHost()) || uri.getUserInfo() != null) {
                throw new IllegalArgumentException();
            }
        } catch (IllegalArgumentException failure) {
            throw new BusinessException(label + "需要可供供应商访问的 HTTP/HTTPS 链接");
        }
        return url;
    }
}
