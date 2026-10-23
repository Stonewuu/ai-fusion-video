package com.stonewu.fusion.service.canvas.capability;

import java.util.List;

/**
 * 前端表单、工具参数与服务端配置校验共用的字段约束。
 */
public record CanvasConfigField(String type, boolean required, Double minimum,
                                Double maximum, Integer maxLength, List<String> enumValues) {
    public CanvasConfigField {
        enumValues = List.copyOf(enumValues);
    }

    public static CanvasConfigField text(int maxLength) {
        return new CanvasConfigField("string", false, null, null, maxLength, List.of());
    }

    public static CanvasConfigField integer(long min, long max, boolean required) {
        return new CanvasConfigField("integer", required, (double) min, (double) max, null, List.of());
    }

    public static CanvasConfigField number(double min, double max) {
        return new CanvasConfigField("number", false, min, max, null, List.of());
    }

    public static CanvasConfigField flag() {
        return new CanvasConfigField("boolean", false, null, null, null, List.of());
    }

    public static CanvasConfigField object() {
        return new CanvasConfigField("object", false, null, null, 128000, List.of());
    }

    public static CanvasConfigField stringList(int maxItems, int itemLength) {
        return new CanvasConfigField("string_array", false, 0d, (double) maxItems, itemLength, List.of());
    }

    public static CanvasConfigField choice(boolean required, String... values) {
        return new CanvasConfigField("string", required, null, null, 128, List.of(values));
    }
}
