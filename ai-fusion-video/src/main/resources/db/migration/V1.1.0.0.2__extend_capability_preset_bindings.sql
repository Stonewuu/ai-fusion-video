-- 为 V1.1.0.0.1 发布后新增的内置模型补充能力预设绑定。
UPDATE `afv_ai_model`
SET `capability_preset_code` = `code`
WHERE `code` IN (
  'doubao-seedance-2-0-mini-260615',
  'doubao-seedream-5-0-pro-260628',
  'sora-2',
  'sora-2-pro'
);

-- 可灵部分图片/视频模型复用同一上游 model code，能力预设代码必须按模型类型区分。
UPDATE `afv_ai_model`
SET `capability_preset_code` = CASE
  WHEN `code` = 'kling-v1-5' AND `model_type` = 2 THEN 'kling-v1-5-image'
  WHEN `code` = 'kling-v2' AND `model_type` = 2 THEN 'kling-v2-image'
  WHEN `code` = 'kling-v2-1' AND `model_type` = 2 THEN 'kling-v2-1-image'
  WHEN `code` = 'kling-v2-new' AND `model_type` = 2 THEN 'kling-v2-new-image'
  WHEN `code` = 'kling-image-o1' AND `model_type` = 2 THEN 'kling-image-o1-image'
  WHEN `code` = 'kling-v3' AND `model_type` = 2 THEN 'kling-v3-image'
  WHEN `code` = 'kling-v3-omni' AND `model_type` = 2 THEN 'kling-v3-omni-image'
  WHEN `code` = 'kling-v1-6' AND `model_type` = 3 THEN 'kling-v1-6-video'
  WHEN `code` = 'kling-v2-master' AND `model_type` = 3 THEN 'kling-v2-master-video'
  WHEN `code` = 'kling-v2-1' AND `model_type` = 3 THEN 'kling-v2-1-video'
  WHEN `code` = 'kling-v2-1-master' AND `model_type` = 3 THEN 'kling-v2-1-master-video'
  WHEN `code` = 'kling-v2-5-turbo' AND `model_type` = 3 THEN 'kling-v2-5-turbo-video'
  WHEN `code` = 'kling-v2-6' AND `model_type` = 3 THEN 'kling-v2-6-video'
  WHEN `code` = 'kling-video-o1' AND `model_type` = 3 THEN 'kling-video-o1-video'
  WHEN `code` = 'kling-v3' AND `model_type` = 3 THEN 'kling-v3-video'
  WHEN `code` = 'kling-v3-omni' AND `model_type` = 3 THEN 'kling-v3-omni-video'
  WHEN `code` = 'kling-3.0-turbo' AND `model_type` = 3 THEN 'kling-3-0-turbo-video'
  ELSE `capability_preset_code`
END
WHERE (`model_type` = 2 AND `code` IN (
  'kling-v1-5', 'kling-v2', 'kling-v2-1', 'kling-v2-new',
  'kling-image-o1', 'kling-v3', 'kling-v3-omni'
)) OR (`model_type` = 3 AND `code` IN (
  'kling-v1-6', 'kling-v2-master', 'kling-v2-1', 'kling-v2-1-master',
  'kling-v2-5-turbo', 'kling-v2-6', 'kling-video-o1', 'kling-v3',
  'kling-v3-omni', 'kling-3.0-turbo'
));
