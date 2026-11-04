package com.stonewu.fusion.service.ai.model;

public final class GptImageModel {
    private GptImageModel() {
    }

    /** 路由前缀只用于选择渠道，能力按具体型号匹配。发送请求时使用用户保存的完整标识。 */
    public static String presetCode(String modelCode) {
        if (modelCode == null) {
            return null;
        }
        String code = modelCode.trim();
        return switch (code.substring(code.lastIndexOf('/') + 1)) {
            case "gpt-image-2", "gpt-image-2-2026-04-21" -> "gpt-image-2";
            case "gpt-image-2.5-sunburst", "gpt-image-2.5-sunburst-2026-09-08" -> "gpt-image-2.5-sunburst";
            case "gpt-image-2.5-flare", "gpt-image-2.5-flare-2026-09-08" -> "gpt-image-2.5-flare";
            default -> null;
        };
    }
}
