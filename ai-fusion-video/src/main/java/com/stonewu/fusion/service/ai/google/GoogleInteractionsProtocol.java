package com.stonewu.fusion.service.ai.google;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;

/** 两个渠道统一使用输入步骤；Vertex v1 的输出格式使用数组。 */
public final class GoogleInteractionsProtocol {
    private GoogleInteractionsProtocol() {
    }

    public static JSONArray input(JSONArray content) {
        return new JSONArray().put(new JSONObject().set("type", "user_input").set("content", content));
    }

    public static Object format(JSONObject format, boolean developer) {
        return developer ? format : new JSONArray().put(format);
    }
}
