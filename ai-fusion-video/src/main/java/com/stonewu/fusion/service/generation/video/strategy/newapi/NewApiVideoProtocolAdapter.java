package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;

/**
 * NewAPI 视频协议适配器。
 */
public interface NewApiVideoProtocolAdapter {

    String DEFAULT_SUBMIT_PATH = "/v1/video/generations";

    String getProtocol();

    JSONObject buildSubmitBody(NewApiVideoProtocolContext context);

    default String resolveSubmitPath(NewApiVideoProtocolContext context) {
        return DEFAULT_SUBMIT_PATH;
    }

    default String resolveQueryPath(NewApiVideoProtocolContext context, String taskId) {
        return resolveSubmitPath(context) + "/" + taskId;
    }
}
