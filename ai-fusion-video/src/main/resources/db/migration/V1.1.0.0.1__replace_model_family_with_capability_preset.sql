ALTER TABLE `afv_ai_model`
  ADD COLUMN `capability_preset_code` VARCHAR(128) NULL DEFAULT NULL
    COMMENT '模型能力预设代码；NULL 表示自定义能力配置'
    AFTER `model_protocol`;

-- 只有与内置预设代码精确匹配的现有模型才绑定能力预设，禁止按系列、协议或名称猜测。
UPDATE `afv_ai_model`
SET `capability_preset_code` = `code`
WHERE `code` IN (
  'agnes-image-2.0-flash',
  'agnes-image-2.1-flash',
  'agnes-video-v2.0',
  'dall-e-3',
  'doubao-seedance-1-0-lite-i2v-250428',
  'doubao-seedance-1-0-lite-t2v-250428',
  'doubao-seedance-1-0-pro-250528',
  'doubao-seedance-1-0-pro-fast-251015',
  'doubao-seedance-1-5-pro-251215',
  'doubao-seedance-2-0-260128',
  'doubao-seedance-2-0-fast-260128',
  'doubao-seedream-3-0-t2i-250415',
  'doubao-seedream-4-0-250828',
  'doubao-seedream-4-5-251128',
  'doubao-seedream-5-0-260128',
  'doubao-seedream-5-0-lite-260128',
  'gemini-2.5-flash-image',
  'gemini-3.0-pro-image',
  'gemini-3.1-flash-image',
  'gpt-image-1',
  'gpt-image-1.5',
  'gpt-image-1-mini',
  'gpt-image-2',
  'imagen-3.0-generate-002',
  'imagen-4.0-generate-preview',
  'qwen-image',
  'qwen-image-2.0',
  'qwen-image-2.0-pro',
  'qwen-image-max',
  'qwen-image-plus',
  'veo_3_1_i2v_lite',
  'veo_3_1_i2v_s',
  'veo_3_1_i2v_s_fast_fl',
  'veo_3_1_i2v_s_fast_ultra_fl',
  'veo_3_1_i2v_s_fast_ultra_relaxed',
  'veo_3_1_interpolation_lite',
  'veo_3_1_r2v_fast',
  'veo_3_1_r2v_fast_ultra',
  'veo_3_1_r2v_fast_ultra_relaxed',
  'veo_3_1_t2v',
  'veo_3_1_t2v_fast',
  'veo_3_1_t2v_fast_ultra',
  'veo_3_1_t2v_fast_ultra_relaxed',
  'veo_3_1_t2v_lite',
  'wan2.2-kf2v-flash',
  'wan2.6-i2v',
  'wan2.6-i2v-flash',
  'wan2.6-t2v',
  'wan2.7-i2v',
  'wan2.7-image',
  'wan2.7-image-pro',
  'wan2.7-r2v',
  'wan2.7-t2v',
  'wan2.7-videoedit',
  'wanx2.1-i2v-plus',
  'wanx2.1-i2v-turbo',
  'wanx2.1-kf2v-plus',
  'wanx2.1-t2v-plus',
  'wanx2.1-t2v-turbo'
);

ALTER TABLE `afv_ai_model`
  DROP COLUMN `model_family`;
