package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * NewAPI Sora 协议适配器。
 * 当前先复用通用视频组包，后续 Sora 字段分叉时只需在此类扩展。
 */
@Component
@RequiredArgsConstructor
public class NewApiSoraVideoProtocolAdapter implements NewApiVideoProtocolAdapter {

    private final NewApiVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return "sora";
    }

    @Override
    public JSONObject buildSubmitBody(NewApiVideoProtocolContext context) {
        return support.buildGenericSubmitBody(context);
    }
}
