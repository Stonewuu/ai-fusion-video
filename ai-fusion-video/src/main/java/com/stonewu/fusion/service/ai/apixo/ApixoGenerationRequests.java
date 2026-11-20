package com.stonewu.fusion.service.ai.apixo;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;

import java.util.ArrayList;

/** 将创作工作区的输入映射到 APIXO 模型目录声明的参数和模式。 */
public final class ApixoGenerationRequests {
    private ApixoGenerationRequests() {
    }

    public static JSONObject image(ImageTask task, AiModel model) {
        var contract = ApixoInputContract.from(model);
        var refs = ApixoInputContract.sources(task.getRefImageUrls());
        var input = contract.input(refs.isEmpty() ? "textMode" : "imageMode", task.getPrompt());
        var config = contract.config();
        if (task.getWidth() != null || task.getHeight() != null) {
            if (!config.getBool("supportCustomSize", false)) {
                throw new BusinessException("此 APIXO 模型请使用分辨率和画面比例选择图片尺寸");
            }
            Integer width = task.getWidth();
            Integer height = task.getHeight();
            if (width == null || height == null || width <= 0 || height <= 0) {
                throw new BusinessException("请同时填写有效的图片宽度和高度");
            }
            if (notBlank(task.getResolution()) || notBlank(task.getRatio())) {
                throw new BusinessException("请在图片宽高与分辨率档位中选择一种尺寸方式");
            }
            long pixels = (long) width * height;
            if (pixels < config.getLong("minPixels") || pixels > config.getLong("maxPixels")
                    || (double) Math.max(width, height) / Math.min(width, height) > config.getDouble("maxAspectRatio")) {
                throw new BusinessException("图片尺寸超出此 APIXO 模型支持的像素或比例范围");
            }
            contract.put(input, "size", width + "x" + height);
        } else {
            contract.put(input, "resolution", notBlank(task.getResolution()) ? task.getResolution()
                    : config.getStr("defaultResolution"));
            String ratio = notBlank(task.getRatio()) ? task.getRatio()
                    : notBlank(task.getAspectRatio()) ? task.getAspectRatio() : config.getStr("defaultAspectRatio");
            contract.put(input, "aspect_ratio", contract.ratio(ratio));
        }
        contract.references(input, "image_urls", refs);
        contract.validate(input);
        return input;
    }

    public static JSONObject video(VideoTask task, AiModel model) {
        var contract = ApixoInputContract.from(model);
        var config = contract.config();
        var refs = ApixoInputContract.sources(task.getReferenceImageUrls());
        var videos = ApixoInputContract.sources(task.getReferenceVideoUrls());
        var audios = ApixoInputContract.sources(task.getReferenceAudioUrls());
        boolean first = notBlank(task.getFirstFrameImageUrl());
        boolean last = notBlank(task.getLastFrameImageUrl());
        boolean references = !refs.isEmpty() || !videos.isEmpty() || !audios.isEmpty();
        if (last && !first) {
            throw new BusinessException("使用尾帧时请同时提供首帧");
        }
        if ((first || last) && references) {
            throw new BusinessException("此 APIXO 模型的首尾帧和参考素材需要分开使用");
        }
        var images = new ArrayList<>(refs);
        if (first) {
            images.add(task.getFirstFrameImageUrl());
        }
        if (last) {
            images.add(task.getLastFrameImageUrl());
        }
        if ((first || last) && images.size() > contract.definition().getInt("maxFrameImages")) {
            throw new BusinessException("此 APIXO 模型不支持当前首尾帧组合");
        }
        var input = contract.input(first ? "frameMode" : references ? "referenceMode" : "textMode", task.getPrompt());
        contract.put(input, "resolution", notBlank(task.getResolution()) ? task.getResolution()
                : config.getStr("defaultResolution"));
        contract.put(input, "aspect_ratio", contract.ratio(notBlank(task.getRatio()) ? task.getRatio()
                : config.getStr("defaultAspectRatio")));
        Integer duration = task.getDuration() == null ? config.getInt("defaultDuration") : task.getDuration();
        Integer referenceDuration = contract.definition().getInt("referenceDuration");
        if (references && referenceDuration != null && !referenceDuration.equals(duration)) {
            throw new BusinessException("此 APIXO 模型使用参考素材时的视频时长须为 " + referenceDuration + " 秒");
        }
        contract.put(input, "duration", duration);
        contract.put(input, "seed", task.getSeed());
        if (contract.fields().containsKey("sound")) {
            contract.put(input, "sound", task.getGenerateAudio());
        }
        if (task.getWatermark() != null && config.getBool("supportWatermark", false)) {
            if (contract.fields().containsKey("remove_watermark")) {
                contract.put(input, "remove_watermark", !task.getWatermark());
            } else {
                contract.put(input, "watermark", task.getWatermark());
            }
        }
        contract.references(input, "image_urls", images);
        contract.references(input, "video_urls", videos);
        contract.references(input, "audio_urls", audios);
        contract.validate(input);
        return input;
    }

    private static boolean notBlank(String value) {
        return value != null && !value.isBlank();
    }
}
