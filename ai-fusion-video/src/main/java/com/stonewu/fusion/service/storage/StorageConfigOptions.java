package com.stonewu.fusion.service.storage;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;

import java.util.Collections;
import java.util.Map;

/**
 * JSON option helpers for storage provider-specific settings.
 */
public final class StorageConfigOptions {

    private StorageConfigOptions() {
    }

    public static JSONObject parseOrEmpty(String optionsJson) {
        if (StrUtil.isBlank(optionsJson)) {
            return new JSONObject();
        }
        try {
            return JSONUtil.parseObj(optionsJson);
        } catch (Exception ignored) {
            return new JSONObject();
        }
    }

    public static String toJson(Map<String, Object> options) {
        if (options == null || options.isEmpty()) {
            return null;
        }
        return JSONUtil.toJsonStr(options);
    }

    public static Map<String, Object> toMap(String optionsJson) {
        JSONObject object = parseOrEmpty(optionsJson);
        if (object.isEmpty()) {
            return Collections.emptyMap();
        }
        return object;
    }

    public static boolean getBoolean(JSONObject options, String key, boolean defaultValue) {
        if (options == null || !options.containsKey(key)) {
            return defaultValue;
        }
        Boolean value = options.getBool(key);
        return value != null ? value : defaultValue;
    }

    public static String getString(JSONObject options, String key) {
        if (options == null || !options.containsKey(key)) {
            return null;
        }
        String value = options.getStr(key);
        return StrUtil.isBlank(value) ? null : value;
    }
}
