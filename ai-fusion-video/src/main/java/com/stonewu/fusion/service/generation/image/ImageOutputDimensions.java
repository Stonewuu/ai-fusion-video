package com.stonewu.fusion.service.generation.image;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.ImageTask;

import java.util.regex.Pattern;

/** 保留用户选的档位和比例。有像素映射时同时记录宽高，供需要像素尺寸的协议使用。 */
public final class ImageOutputDimensions {
    private static final Pattern PIXELS = Pattern.compile("([1-9]\\d{0,5})[xX×*]([1-9]\\d{0,5})");
    private static final Pattern RATIO = Pattern.compile("([1-9]\\d{0,4}):([1-9]\\d{0,4})");

    private ImageOutputDimensions() {
    }

    public static void normalize(ImageTask task, JSONObject config, boolean gptImage) {
        String resolution = StrUtil.trimToEmpty(task.getResolution());
        String ratio = StrUtil.trim(StrUtil.blankToDefault(task.getRatio(), task.getAspectRatio()));
        JSONObject sizes = config.getJSONObject("supportedSizes");
        boolean hasTable = sizes != null && !sizes.isEmpty();
        if (!hasTable && !gptImage) {
            return;
        }
        if (StrUtil.isNotBlank(task.getAspectRatio()) && !task.getAspectRatio().trim().equals(ratio)) {
            throw new BusinessException(400, "两处填写的画面比例不一致");
        }
        Integer width = task.getWidth();
        Integer height = task.getHeight();
        if (gptImage && (resolution.equals("auto") || resolution.isEmpty() && "auto".equals(ratio) && width == null && height == null)) {
            if (!Boolean.TRUE.equals(config.getBool("supportAutoSize")) || width != null || height != null
                    || StrUtil.isNotBlank(ratio) && !"auto".equals(ratio)) {
                throw new BusinessException(400, "自动分辨率需要使用自动比例，并留空宽高");
            }
            task.setResolution("auto");
            task.setRatio("auto");
            task.setAspectRatio("auto");
            return;
        }
        if (resolution.isEmpty() && gptImage && width == null && height == null) {
            resolution = StrUtil.blankToDefault(config.getStr("defaultSizeTier"), config.getStr("defaultResolution", ""));
            if (resolution.isEmpty() && StrUtil.isNotBlank(ratio) && !ratio.equals(config.getStr("defaultAspectRatio", "1:1"))) {
                throw new BusinessException(400, "请为当前模型配置该比例的预设尺寸，或指定图片宽高");
            }
        }
        String pixelSize = resolution;
        JSONObject tier = hasTable ? sizes.getJSONObject(resolution) : null;
        if (tier != null) {
            if (StrUtil.isBlank(ratio)) {
                ratio = config.getStr("defaultAspectRatio", tier.containsKey("1:1") ? "1:1" : null);
            }
            pixelSize = StrUtil.isNotBlank(ratio) ? tier.getStr(ratio) : null;
            if (StrUtil.isBlank(pixelSize)) {
                throw new BusinessException(400, "当前分辨率档位与画面比例不匹配，请重新选择");
            }
        }
        var pixels = PIXELS.matcher(StrUtil.trimToEmpty(pixelSize));
        if (pixels.matches()) {
            int w = Integer.parseInt(pixels.group(1));
            int h = Integer.parseInt(pixels.group(2));
            if (width != null && width != w || height != null && height != h) {
                throw new BusinessException(400, "图片宽高与所选分辨率不一致");
            }
            width = w;
            height = h;
        } else if (gptImage && !resolution.isEmpty()) {
            throw new BusinessException(400, "当前图片模型不支持该分辨率档位");
        }
        if (width == null && height == null) {
            return;
        }
        if (width == null || height == null || width <= 0 || height <= 0) {
            throw new BusinessException(400, "请同时填写有效的图片宽度和高度");
        }
        if (gptImage) {
            validateGptSize(width, height);
        }
        var aspect = RATIO.matcher(StrUtil.trimToEmpty(ratio));
        if (gptImage && tier == null && aspect.matches()) {
            int a = Integer.parseInt(aspect.group(1));
            int b = Integer.parseInt(aspect.group(2));
            // 预设可能按模型要求取整到指定像素倍数，如 2048×1360 对应 3:2。
            int step = config.getInt("sizeMultiple", 16);
            if (Math.abs((long) width * b - (long) height * a) > (long) Math.max(a, b) * step / 2) {
                throw new BusinessException(400, "图片宽高与画面比例不一致，请重新选择分辨率");
            }
        }
        task.setWidth(width);
        task.setHeight(height);
        if (tier != null) {
            task.setResolution(resolution);
            task.setRatio(ratio);
            task.setAspectRatio(ratio);
        }
    }

    public static void validateGptSize(int width, int height) {
        long pixels = (long) width * height;
        if (width <= 0 || height <= 0 || width % 16 != 0 || height % 16 != 0
                || Math.max(width, height) > 3840 || Math.max(width, height) > 3L * Math.min(width, height)
                || pixels < 655360 || pixels > 8294400) {
            throw new BusinessException(400, "图片宽高须为 16 的倍数，最长边不超过 3840，长短边之比不超过 3:1，总像素须在 655360 至 8294400 之间");
        }
    }
}
