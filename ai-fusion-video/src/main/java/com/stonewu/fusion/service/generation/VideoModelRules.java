package com.stonewu.fusion.service.generation;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.VideoTask;

import java.util.List;

/**
 * 视频能力契约 v1：纯参数规则，不访问供应商、不猜测模型名称、不改变图结构。
 */
public record VideoModelRules(
        Integer version, String profile, Boolean promptRequired, Integer promptMaxLength,
        Boolean automaticDuration, String defaultRatio, String frameRatio, String frameRatioMode,
        Boolean textRequiresFixedRatio, Boolean fixedGenerateAudio, Boolean defaultGenerateAudio,
        Boolean lastFrameRequiresFirst,
        Integer maxReferenceInputs, Integer maxInputOutputVideoSeconds,
        MediaLimits imageLimits, MediaLimits referenceVideoLimits, MediaLimits referenceAudioLimits,
        List<String> promptExpansionModes, List<String> referenceTaskTypes, List<String> outputFormats,
        Boolean supportPromptExtend) {
    private static final ObjectMapper JSON = new ObjectMapper();

    public record MediaLimits(Double minSeconds, Double maxSeconds, Double maxTotalSeconds,
                              Double singleItemMaxTotalSeconds, Long maxBytes,
                              Integer minDimension, Integer maxDimension, Double minAspect, Double maxAspect,
                              Long minPixels, Long maxPixels, List<String> mimeTypes) {
    }

    public record Inputs(boolean firstFrame, boolean lastFrame, int images, int videos, int audios) {
        public boolean frames() {
            return firstFrame || lastFrame;
        }

        public boolean references() {
            return images > 0 || videos > 0 || audios > 0;
        }

        public boolean any() {
            return frames() || references();
        }

        public int total() {
            return images + videos + audios + (firstFrame ? 1 : 0) + (lastFrame ? 1 : 0);
        }
    }

    public record Parameters(String resolution, String ratio, Integer duration, Long seed,
                             Boolean generateAudio, Boolean watermark, Boolean cameraFixed) {
    }

    public static VideoModelRules from(JSONObject config) {
        var value = config.getJSONObject("videoRules");
        if (value == null) {
            return null; // 已有模型未声明新契约时，继续使用原能力校验。
        }
        try {
            var rules = JSON.readValue(value.toString(), VideoModelRules.class);
            if (!Integer.valueOf(1).equals(rules.version()) || StrUtil.isBlank(rules.profile())) {
                throw new BusinessException("视频模型能力配置版本无效，请重新选择能力预设");
            }
            return rules;
        } catch (JsonProcessingException invalid) {
            throw new BusinessException("视频模型能力配置无效，请检查能力预设与覆盖参数");
        }
    }

    public static Inputs inputs(VideoTask task) {
        return new Inputs(StrUtil.isNotBlank(task.getFirstFrameImageUrl()),
                StrUtil.isNotBlank(task.getLastFrameImageUrl()),
                count(task.getReferenceImageUrls()), count(task.getReferenceVideoUrls()),
                count(task.getReferenceAudioUrls()));
    }

    public static Parameters parameters(VideoTask task) {
        return new Parameters(task.getResolution(), task.getRatio(), task.getDuration(), task.getSeed(),
                task.getGenerateAudio(), task.getWatermark(), task.getCameraFixed());
    }

    public static void normalizeTask(JSONObject config, VideoTask task) {
        var rules = from(config);
        if (rules == null) {
            return;
        }
        var inputs = inputs(task);
        rules.validatePrompt(task.getPrompt(), inputs.any());
        var result = rules.resolve(config, parameters(task), inputs);
        task.setResolution(result.resolution());
        task.setRatio(result.ratio());
        task.setDuration(result.duration());
        task.setGenerateAudio(result.generateAudio());
        task.setWatermark(result.watermark());
        task.setCameraFixed(result.cameraFixed());
        if (task.getPrompt() == null) {
            task.setPrompt("");
        }
    }

    public void validatePrompt(String prompt, boolean hasMedia) {
        if (StrUtil.isBlank(prompt) && (Boolean.TRUE.equals(promptRequired) || !hasMedia)) {
            throw new BusinessException(
                    Boolean.TRUE.equals(promptRequired) ? "此模型需要填写创作要求" : "请填写创作要求或添加参考素材");
        }
        if (prompt != null && promptMaxLength != null && prompt.codePointCount(0, prompt.length()) > promptMaxLength) {
            throw new BusinessException("此模型的创作要求最多允许 " + promptMaxLength + " 个字符");
        }
    }

    public Parameters resolve(JSONObject config, Parameters supplied, Inputs inputs) {
        if (Boolean.TRUE.equals(lastFrameRequiresFirst) && inputs.lastFrame() && !inputs.firstFrame()) {
            throw new BusinessException("使用尾帧时，请同时添加首帧图片");
        }
        if (Boolean.TRUE.equals(config.getBool("exclusiveInputModes")) && inputs.frames() && inputs.references()) {
            throw new BusinessException("首尾帧与参考图片、视频、音频不能混用，请选择一种输入方式");
        }
        if (inputs.frames() && inputs.videos() > 0 && Boolean.FALSE.equals(config.getBool("framesWithReferenceVideos"))) {
            throw new BusinessException("当前模型的首尾帧不能与参考视频混用");
        }
        if (maxReferenceInputs != null && inputs.total() > maxReferenceInputs) {
            throw new BusinessException("参考素材总数不能超过 " + maxReferenceInputs + " 项");
        }
        if (Boolean.TRUE.equals(config.getBool("referenceAudioRequiresVisualInput")) && inputs.audios() > 0
                && !inputs.frames() && inputs.images() == 0 && inputs.videos() == 0) {
            throw new BusinessException("此模型的参考音频需要搭配图片或视频");
        }

        String resolution = choice(config, "supportedResolutions",
                StrUtil.blankToDefault(supplied.resolution(), config.getStr("defaultResolution")), "分辨率");
        String ratio = choice(config, "supportedAspectRatios",
                StrUtil.blankToDefault(supplied.ratio(), defaultRatio), "画面比例");
        if (inputs.frames() && StrUtil.isNotBlank(frameRatio)) {
            if (StrUtil.isNotBlank(supplied.ratio()) && !frameRatio.equals(ratio) && !"inferred".equals(
                    frameRatioMode)) {
                throw new BusinessException("首尾帧生成的画面比例须选择“自适应”");
            }
            ratio = frameRatio;
        }
        if (!inputs.any() && Boolean.TRUE.equals(textRequiresFixedRatio) && "adaptive".equals(ratio)) {
            throw new BusinessException("此模型的纯文本生成需要明确画面比例，请选择 16:9、9:16 等比例");
        }

        Integer duration = supplied.duration() == null ? config.getInt("defaultDuration") : supplied.duration();
        if (duration == null) {
            throw new BusinessException("请设置视频时长");
        }
        if (duration == -1) {
            if (!Boolean.TRUE.equals(automaticDuration)) {
                throw new BusinessException("当前模型不支持自动时长");
            }
        } else {
            Integer minimum = config.getInt("minDuration");
            Integer maximum = config.getInt("maxDuration");
            if (duration < 1 || minimum != null && duration < minimum || maximum != null && duration > maximum) {
                throw new BusinessException("视频时长超出模型允许范围：" + minimum + " 至 " + maximum + " 秒");
            }
            var allowed = config.getJSONArray("supportedDurations");
            if (allowed != null && !allowed.toList(Integer.class).contains(duration)) {
                throw new BusinessException("当前模型不支持此视频时长");
            }
            var durationsByResolution = config.getJSONObject("supportedDurationsByResolution");
            if (durationsByResolution != null) {
                var resolutionDurations = durationsByResolution.getJSONArray(resolution);
                if (resolutionDurations != null && !resolutionDurations.toList(Integer.class).contains(duration)) {
                    throw new BusinessException(resolution + " 分辨率不支持此视频时长");
                }
            }
            var resolutionLimits = config.getJSONObject("maxDurationByResolution");
            if (resolutionLimits != null) {
                Integer resolutionMaximum = resolutionLimits.getInt(resolution);
                if (resolutionMaximum != null && duration > resolutionMaximum) {
                    throw new BusinessException(resolution + " 分辨率的视频最长支持 " + resolutionMaximum + " 秒");
                }
            }
        }
        if (supplied.seed() != null) {
            if (Boolean.FALSE.equals(config.getBool("supportSeed"))) {
                throw new BusinessException("当前模型不支持随机种子参数");
            }
            if (supplied.seed() < -1 || supplied.seed() > Integer.MAX_VALUE) {
                throw new BusinessException("随机种子须为 -1 或 0 至 2147483647 的整数");
            }
        }
        Boolean audio = supplied.generateAudio() == null ? defaultGenerateAudio : supplied.generateAudio();
        if (fixedGenerateAudio != null) {
            if (audio != null && !fixedGenerateAudio.equals(audio)) {
                throw new BusinessException("当前模型固定生成带声音的视频，不能关闭声音");
            }
            audio = fixedGenerateAudio;
        } else if (Boolean.TRUE.equals(audio) && Boolean.FALSE.equals(config.getBool("supportGenerateAudio"))) {
            throw new BusinessException("当前模型不支持开启视频声音");
        }
        if (Boolean.TRUE.equals(supplied.watermark()) && Boolean.FALSE.equals(config.getBool("supportWatermark"))) {
            throw new BusinessException("当前模型不支持水印开关");
        }
        if (Boolean.TRUE.equals(supplied.cameraFixed()) && Boolean.FALSE.equals(config.getBool("supportCameraFixed"))) {
            throw new BusinessException("当前模型不支持固定镜头参数");
        }
        validateOptions(config, inputs, ratio, duration);
        return new Parameters(resolution, ratio, duration, supplied.seed(), audio,
                Boolean.FALSE.equals(config.getBool("supportWatermark")) ? null : supplied.watermark(),
                Boolean.FALSE.equals(config.getBool("supportCameraFixed")) ? null : supplied.cameraFixed());
    }

    private void validateOptions(JSONObject config, Inputs inputs, String ratio, int duration) {
        option(config, "promptExpansionMode", promptExpansionModes, "提示词优化方式");
        option(config, "outputFormat", outputFormats, "视频输出格式");
        String taskType = option(config, "referenceTaskType", referenceTaskTypes, "参考任务类型");
        if (taskType != null && !taskType.equals("auto") && !inputs.references()) {
            throw new BusinessException("当前参考任务类型需要添加参考素材");
        }
        if ("edit".equals(taskType) || "extend".equals(taskType)) {
            if (inputs.videos() == 0) {
                throw new BusinessException("编辑或延长视频需要连接参考视频");
            }
            if (!"adaptive".equals(ratio)) {
                throw new BusinessException("编辑或延长视频的画面比例须选择自适应");
            }
            if ("edit".equals(taskType) && duration != -1) {
                throw new BusinessException("编辑视频须使用自动时长，以保留原视频时长");
            }
        }
        if (config.containsKey("promptExtend") && (!Boolean.TRUE.equals(supportPromptExtend) || !(config.get(
                "promptExtend") instanceof Boolean))) {
            throw new BusinessException("此模型的提示词扩写设置无效");
        }
    }

    private static String option(JSONObject config, String key, List<String> supported, String label) {
        if (!config.containsKey(key)) {
            return null;
        }
        if (!(config.get(key) instanceof String value) || supported == null || !supported.contains(value)) {
            throw new BusinessException("此模型不支持当前" + label + "设置");
        }
        return value;
    }

    /**
     * 自动时长使用模型上限作预算，不能以 -1、0 或默认五秒降低授权范围。
     */
    public static int budgetSeconds(JSONObject config, Integer requested) {
        Integer duration = requested == null ? config.getInt("defaultDuration") : requested;
        if (Integer.valueOf(-1).equals(duration)) {
            var rules = from(config);
            Integer maximum = config.getInt("maxDuration");
            if (rules == null || !Boolean.TRUE.equals(rules.automaticDuration()) || maximum == null || maximum <= 0) {
                throw new BusinessException("此模型没有有效的自动时长上限，请指定视频时长");
            }
            return maximum;
        }
        if (duration == null || duration <= 0) {
            throw new BusinessException("请指定视频时长，以便计算运行范围");
        }
        return duration;
    }

    private static String choice(JSONObject config, String key, String value, String label) {
        if (StrUtil.isBlank(value)) {
            throw new BusinessException("请设置" + label);
        }
        var allowed = config.getJSONArray(key);
        if (allowed == null || allowed.isEmpty()) {
            throw new BusinessException("模型未配置可用的" + label);
        }
        return allowed.toList(String.class).stream().filter(item -> item.equalsIgnoreCase(value)).findFirst()
                .orElseThrow(() -> new BusinessException("当前模型不支持此" + label + "：" + value));
    }

    private static int count(String values) {
        if (StrUtil.isBlank(values)) {
            return 0;
        }
        try {
            List<String> items = JSONUtil.parseArray(values).toList(String.class);
            if (items.stream().anyMatch(StrUtil::isBlank)) {
                throw new IllegalArgumentException();
            }
            return items.size();
        } catch (RuntimeException invalid) {
            throw new BusinessException("参考素材列表格式无效");
        }
    }
}
