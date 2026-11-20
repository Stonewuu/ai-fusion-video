package com.stonewu.fusion.service.ai.apixo;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** 在供应商边界按模型目录校验参数，不根据模型名称推断能力。 */
public record ApixoInputContract(JSONObject config, JSONObject definition, JSONObject fields) {
    private static final Map<String, String> LABELS = Map.ofEntries(
            Map.entry("mode", "生成方式"), Map.entry("generationType", "生成方式"),
            Map.entry("prompt", "提示词"), Map.entry("resolution", "分辨率"),
            Map.entry("aspect_ratio", "画面比例"), Map.entry("duration", "视频时长"),
            Map.entry("image_urls", "参考图片"), Map.entry("video_urls", "参考视频"),
            Map.entry("audio_urls", "参考音频"), Map.entry("seed", "随机种子"),
            Map.entry("size", "图片尺寸"), Map.entry("sound", "声音"), Map.entry("watermark", "水印"));

    public static ApixoInputContract from(AiModel model) {
        JSONObject config = JSONUtil.parseObj(model.getConfig());
        JSONObject definition = config.getJSONObject("apixo");
        if (definition == null || definition.getJSONObject("fields") == null
                || definition.getJSONObject("defaults") == null || definition.getJSONObject("ratioValues") == null
                || definition.getStr("modeField") == null) {
            throw new BusinessException("请为此模型选择 APIXO 能力预设");
        }
        return new ApixoInputContract(config, definition, definition.getJSONObject("fields"));
    }

    public JSONObject input(String modeKey, String prompt) {
        String mode = definition.getStr(modeKey);
        if (mode == null) {
            throw new BusinessException("此 APIXO 模型不支持当前素材组合，请调整参考素材");
        }
        var input = JSONUtil.parseObj(definition.getJSONObject("defaults").toString());
        put(input, definition.getStr("modeField"), mode);
        put(input, "prompt", prompt);
        return input;
    }

    public void put(JSONObject input, String field, Object value) {
        if (value == null || value instanceof String text && text.isBlank()) {
            return;
        }
        JSONObject rule = fields.getJSONObject(field);
        if (rule == null) {
            throw invalid(field, "不受此模型支持，请调整生成参数");
        }
        JSONArray choices = rule.getJSONArray("enum");
        if (choices != null && !choices.isEmpty()) {
            Object selected = choices.stream().filter(choice -> choice.toString().equalsIgnoreCase(value.toString()))
                    .findFirst().orElseThrow(() -> invalid(field, "不在此模型的可选范围内"));
            input.set(field, value instanceof Number ? value : selected);
        } else {
            input.set(field, value);
        }
    }

    public void references(JSONObject input, String field, List<String> values) {
        if (values.isEmpty()) {
            return;
        }
        for (String value : values) {
            try {
                URI uri = URI.create(value);
                if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null || uri.getUserInfo() != null) {
                    throw invalid(field, "需要可访问的 HTTPS 地址，请检查素材存储配置");
                }
            } catch (IllegalArgumentException invalid) {
                throw invalid(field, "地址无效，请重新选择素材");
            }
        }
        put(input, field, values);
    }

    public String ratio(String ratio) {
        var aliases = definition.getJSONObject("ratioValues");
        return aliases.containsKey(ratio) ? aliases.getStr(ratio) : ratio;
    }

    public void validate(JSONObject input) {
        for (String field : input.keySet()) {
            if (!fields.containsKey(field)) {
                throw invalid(field, "不受此模型支持，请检查模型配置");
            }
        }
        for (String field : fields.keySet()) {
            JSONObject rule = fields.getJSONObject(field);
            Object value = input.get(field);
            if (value == null) {
                if (rule.getBool("required", false)) {
                    throw invalid(field, "不能为空，请补充生成参数");
                }
                continue;
            }
            var limits = rule.getJSONObject("constraints");
            if (limits == null) {
                continue;
            }
            if (value instanceof String text) {
                int length = text.codePointCount(0, text.length());
                bounds(field, length, limits.getDouble("minLength"), limits.getDouble("maxLength"));
            } else if (value instanceof Number number) {
                if (number.doubleValue() == -1 && config.getJSONObject("videoRules") != null
                        && config.getJSONObject("videoRules").getBool("automaticDuration", false)) {
                    continue;
                }
                bounds(field, number.doubleValue(), limits.getDouble("minimum"), limits.getDouble("maximum"));
            } else if (value instanceof List<?> list) {
                bounds(field, list.size(), limits.getDouble("minItems"), limits.getDouble("maxItems"));
            }
        }
    }

    public static List<String> sources(String value) {
        if (value == null || value.isBlank()) {
            return List.of();
        }
        if (!JSONUtil.isTypeJSONArray(value)) {
            throw new BusinessException("参考素材列表无效，请重新选择素材");
        }
        List<String> result = new ArrayList<>();
        for (Object source : JSONUtil.parseArray(value)) {
            if (!(source instanceof String text) || text.isBlank()) {
                throw new BusinessException("参考素材列表无效，请重新选择素材");
            }
            result.add(text);
        }
        return List.copyOf(result);
    }

    private static void bounds(String field, double value, Double minimum, Double maximum) {
        if (minimum != null && value < minimum || maximum != null && value > maximum) {
            throw invalid(field, "超出此模型允许的范围，请调整后重试");
        }
    }

    private static BusinessException invalid(String field, String detail) {
        return new BusinessException("APIXO " + LABELS.getOrDefault(field, "生成参数") + detail);
    }
}
