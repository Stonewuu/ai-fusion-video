package com.stonewu.fusion.service.generation.transport;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.VideoTask;

/**
 * Seedance 与 MiniMax 官方 content 元素共用映射；保留各类参考素材的原始顺序。
 */
public final class OfficialVideoContent {
    private OfficialVideoContent() {
    }

    public static JSONArray of(VideoTask task) {
        var content = new JSONArray();
        if (StrUtil.isNotBlank(task.getPrompt())) {
            content.add(new JSONObject().set("type", "text").set("text", task.getPrompt()));
        }
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            content.add(item("image_url", task.getFirstFrameImageUrl(), "first_frame"));
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            content.add(item("image_url", task.getLastFrameImageUrl(), "last_frame"));
        }
        add(content, task.getReferenceImageUrls(), "image_url", "reference_image");
        add(content, task.getReferenceVideoUrls(), "video_url", "reference_video");
        add(content, task.getReferenceAudioUrls(), "audio_url", "reference_audio");
        return content;
    }

    private static void add(JSONArray content, String encoded, String type, String role) {
        if (StrUtil.isBlank(encoded)) {
            return;
        }
        for (String url : JSONUtil.parseArray(encoded).toList(String.class)) {
            if (StrUtil.isBlank(url)) {
                throw new BusinessException("参考素材地址不能为空");
            }
            content.add(item(type, url, role));
        }
    }

    private static JSONObject item(String type, String url, String role) {
        return new JSONObject().set("type", type).set(type, new JSONObject().set("url", url)).set("role", role);
    }
}
