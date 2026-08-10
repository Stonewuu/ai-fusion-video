package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * NewAPI 通用视频协议适配器。
 */
@Component
@RequiredArgsConstructor
public class NewApiGenericVideoProtocolAdapter implements NewApiVideoProtocolAdapter {

    private final NewApiVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return "newapi";
    }

    @Override
    public JSONObject buildSubmitBody(NewApiVideoProtocolContext context) {
        return support.buildGenericSubmitBody(context);
    }
}
