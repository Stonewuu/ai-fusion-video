package com.stonewu.fusion.service.generation;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;

import java.util.List;

/**
 * 使用已经探测且经过访问校验的媒体信息；不为校验重新下载外部素材。
 */
public final class VideoMediaRules {
    private VideoMediaRules() {
    }

    public static void validate(JSONObject config, Integer duration, List<MediaFacts> images,
                                List<MediaFacts> videos, List<MediaFacts> audios) {
        var rules = VideoModelRules.from(config);
        if (rules == null) {
            return;
        }
        validate(rules.imageLimits(), images, MediaKind.IMAGE, "图片");
        double videoSeconds = validate(rules.referenceVideoLimits(), videos, MediaKind.VIDEO, "参考视频");
        validate(rules.referenceAudioLimits(), audios, MediaKind.AUDIO, "参考音频");
        if ("edit".equals(config.getStr("referenceTaskType")) && videos.stream()
                .anyMatch(facts -> facts.durationMs() == null || facts.durationMs() < 4000)) {
            throw new BusinessException("编辑视频的参考视频不能短于 4 秒");
        }
        if (rules.maxInputOutputVideoSeconds() != null && duration != null && duration > 0
                && videoSeconds + duration > rules.maxInputOutputVideoSeconds()) {
            throw new BusinessException(
                    "参考视频与生成视频的时长合计不能超过 " + rules.maxInputOutputVideoSeconds() + " 秒");
        }
    }

    private static double validate(VideoModelRules.MediaLimits limits, List<MediaFacts> values, MediaKind kind,
                                   String label) {
        double total = 0;
        for (var facts : values) {
            if (facts.kind() != kind) {
                throw new BusinessException(label + "的媒体类型不匹配");
            }
            if (limits == null) {
                continue;
            }
            if (limits.maxBytes() != null && facts.byteSize() > limits.maxBytes()) {
                throw new BusinessException(label + "文件过大，请压缩后重新上传");
            }
            if (limits.mimeTypes() != null && !limits.mimeTypes().contains(facts.mimeType())) {
                throw new BusinessException(label + "格式不受此模型支持，请转换格式后重新上传");
            }
            if (limits.minSeconds() != null || limits.maxSeconds() != null || limits.maxTotalSeconds() != null) {
                if (facts.durationMs() == null) {
                    throw new BusinessException("无法确认" + label + "时长，请重新上传素材");
                }
                double seconds = facts.durationMs() / 1000.0;
                if (limits.minSeconds() != null && seconds < limits.minSeconds() || limits.maxSeconds() != null && seconds > limits.maxSeconds()) {
                    throw new BusinessException(
                            label + "每段时长须为 " + limits.minSeconds() + "–" + limits.maxSeconds() + " 秒");
                }
                total += seconds;
            }
            if (limits.minDimension() != null || limits.maxDimension() != null) {
                Integer w = facts.width(), h = facts.height();
                if (w == null || h == null || w <= 0 || h <= 0) {
                    throw new BusinessException("无法确认" + label + "尺寸，请重新上传素材");
                }
                if (limits.minDimension() != null && Math.min(w, h) < limits.minDimension()
                        || limits.maxDimension() != null && Math.max(w, h) > limits.maxDimension()) {
                    throw new BusinessException(
                            label + "宽高须在 " + limits.minDimension() + "–" + limits.maxDimension() + " 像素范围内");
                }
                double aspect = (double) w / h;
                if (limits.minAspect() != null && aspect < limits.minAspect() || limits.maxAspect() != null && aspect > limits.maxAspect()) {
                    throw new BusinessException(label + "宽高比不受此模型支持，请调整画面比例");
                }
                long pixels = (long) w * h;
                if (limits.minPixels() != null && pixels < limits.minPixels() || limits.maxPixels() != null && pixels > limits.maxPixels()) {
                    throw new BusinessException(label + "总像素数不受此模型支持，请调整分辨率");
                }
            }
        }
        if (limits != null) {
            Double maximum = values.size() == 1 && limits.singleItemMaxTotalSeconds() != null
                    ? limits.singleItemMaxTotalSeconds() : limits.maxTotalSeconds();
            if (maximum != null && total > maximum) {
                throw new BusinessException(label + "总时长不能超过 " + maximum + " 秒");
            }
        }
        return total;
    }
}
