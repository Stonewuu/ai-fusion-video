package com.stonewu.fusion.service.ai.google;

import com.stonewu.fusion.common.BusinessException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Pattern;

/** 将 Google 返回的裸 PCM 包装成可播放的 WAV，不改变采样内容。 */
public final class GoogleAudio {
    private static final Pattern RATE = Pattern.compile("rate=(\\d+)");

    private GoogleAudio() {
    }

    public static byte[] wave(byte[] pcm, String mime) {
        var matcher = RATE.matcher(mime);
        int rate = matcher.find() ? Integer.parseInt(matcher.group(1)) : 24000;
        return wave(pcm, rate, 1);
    }

    public static byte[] wave(byte[] pcm, int rate, int channels) {
        if (channels < 1 || channels > 2 || pcm.length % (channels * 2) != 0 || rate < 8000 || rate > 96000) {
            throw new BusinessException("音频采样格式无法读取");
        }
        var output = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
        output.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + pcm.length)
                .put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1)
                .putShort((short) channels).putInt(rate).putInt(rate * channels * 2)
                .putShort((short) (channels * 2)).putShort((short) 16)
                .put("data".getBytes(StandardCharsets.US_ASCII)).putInt(pcm.length).put(pcm);
        return output.array();
    }
}
