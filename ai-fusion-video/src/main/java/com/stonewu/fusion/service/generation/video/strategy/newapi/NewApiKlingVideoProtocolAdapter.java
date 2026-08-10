package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.generation.VideoTask;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * NewAPI 可灵协议适配器。
 * 当前先复用通用视频组包，后续可灵字段分叉时只需在此类扩展。
 */
@Component
@RequiredArgsConstructor
public class NewApiKlingVideoProtocolAdapter implements NewApiVideoProtocolAdapter {

    private final NewApiVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return "kling";
    }

    @Override
    public JSONObject buildSubmitBody(NewApiVideoProtocolContext context) {
        return support.buildGenericSubmitBody(context);
    }

    @Override
    public String resolveSubmitPath(NewApiVideoProtocolContext context) {
        return hasImageInput(context != null ? context.task() : null)
                ? "/kling/v1/videos/image2video"
                : "/kling/v1/videos/text2video";
    }

    private boolean hasImageInput(VideoTask task) {
        return task != null && (StrUtil.isNotBlank(task.getFirstFrameImageUrl())
                || StrUtil.isNotBlank(task.getReferenceImageUrls()));
    }
}
