package com.stonewu.fusion.enums.ai;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * AI 模型类型枚举
 */
@Getter
@AllArgsConstructor
public enum AiModelTypeEnum {

    CHAT(1, "对话模型"),
    IMAGE(2, "图像模型"),
    VIDEO(3, "视频模型"),
    TTS(4, "语音合成"),
    STT(5, "语音识别");

    private final Integer type;
    private final String name;

    public static AiModelTypeEnum valueOf(Integer type) {
        for (AiModelTypeEnum value : values()) {
            if (value.getType().equals(type)) {
                return value;
            }
        }
        return null;
    }
}
