package com.stonewu.fusion.service.ai.google;

import java.util.Locale;

/** Google 目录的类型识别只用于发现，不替换用户保存的调用标识。 */
public final class GoogleModelFamily {
    private GoogleModelFamily() {
    }

    public static Integer type(String code) {
        String name = code.toLowerCase(Locale.ROOT);
        if (excluded(name)) {
            return null;
        }
        if (name.startsWith("lyria-") || name.contains("tts")) {
            return 4;
        }
        if (name.startsWith("gemini-omni-") || name.startsWith("veo-")) {
            return 3;
        }
        if (name.startsWith("imagen-") || name.startsWith("gemini-")
                && (name.contains("image") || name.contains("nano-banana"))) {
            return 2;
        }
        return name.startsWith("gemini-") || name.startsWith("gemma-") ? 1 : null;
    }

    public static boolean excluded(String code) {
        String name = code.toLowerCase(Locale.ROOT);
        return !name.startsWith("lyria-") && (name.startsWith("deep-research-") || name.contains("embedding")
                || name.contains("transcribe") || name.contains("whisper") || name.contains("live")
                || name.contains("realtime") || name.contains("native-audio"));
    }

    public static boolean googlePlatform(String platform) {
        return "gemini".equals(platform) || "vertex_ai".equals(platform) || "vertex_ai_api_key".equals(platform);
    }
}
