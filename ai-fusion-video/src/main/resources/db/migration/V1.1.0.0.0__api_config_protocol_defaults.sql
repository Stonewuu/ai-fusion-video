ALTER TABLE `afv_api_config`
  ADD COLUMN `text_protocol` VARCHAR(64) NULL DEFAULT NULL COMMENT '文本模型默认请求协议' AFTER `platform`,
  ADD COLUMN `image_protocol` VARCHAR(64) NULL DEFAULT NULL COMMENT '图片模型默认请求协议' AFTER `text_protocol`,
  ADD COLUMN `video_protocol` VARCHAR(64) NULL DEFAULT NULL COMMENT '视频模型默认请求协议' AFTER `image_protocol`;

-- 把已知渠道迁移为显式的分能力协议配置；运行时不再根据平台、模型名或家族猜测协议。
UPDATE `afv_api_config`
SET `text_protocol` = CASE LOWER(COALESCE(`platform`, ''))
  WHEN 'openai' THEN 'openai_compatible'
  WHEN 'googleflowreverseapi' THEN 'openai_compatible'
  ELSE LOWER(NULLIF(`platform`, ''))
END
WHERE `text_protocol` IS NULL OR `text_protocol` = '';

UPDATE `afv_api_config`
SET
  `image_protocol` = CASE
    WHEN LOWER(COALESCE(`api_url`, '')) LIKE '%agnes-ai.com%' THEN 'agnes'
    WHEN LOWER(COALESCE(`platform`, '')) IN ('openai', 'openai_compatible')
      AND (`api_url` IS NULL OR `api_url` = '' OR LOWER(`api_url`) LIKE '%api.openai.com%') THEN 'openai'
    WHEN LOWER(COALESCE(`platform`, '')) = 'newapi' THEN 'newapi'
    WHEN LOWER(COALESCE(`platform`, '')) = 'dashscope' THEN 'dashscope'
    WHEN LOWER(COALESCE(`platform`, '')) = 'volcengine' THEN 'volcengine'
    WHEN LOWER(COALESCE(`platform`, '')) IN ('vertex_ai', 'vertexai') THEN 'vertex_ai'
    WHEN LOWER(COALESCE(`platform`, '')) = 'googleflowreverseapi' THEN 'google_flow'
    ELSE NULL
  END,
  `video_protocol` = CASE
    WHEN LOWER(COALESCE(`api_url`, '')) LIKE '%agnes-ai.com%' THEN 'agnes'
    WHEN LOWER(COALESCE(`platform`, '')) IN ('openai', 'openai_compatible')
      AND (`api_url` IS NULL OR `api_url` = '' OR LOWER(`api_url`) LIKE '%api.openai.com%') THEN 'openai'
    WHEN LOWER(COALESCE(`platform`, '')) = 'newapi' THEN 'newapi'
    WHEN LOWER(COALESCE(`platform`, '')) = 'dashscope' THEN 'dashscope'
    WHEN LOWER(COALESCE(`platform`, '')) = 'volcengine' THEN 'volcengine'
    WHEN LOWER(COALESCE(`platform`, '')) = 'googleflowreverseapi' THEN 'google_flow'
    ELSE NULL
  END
WHERE (`image_protocol` IS NULL OR `image_protocol` = '')
   OR (`video_protocol` IS NULL OR `video_protocol` = '');

UPDATE `afv_ai_model`
SET `model_protocol` = CASE LOWER(COALESCE(`model_protocol`, ''))
  WHEN 'generic' THEN NULL
  WHEN 'openai_compatible' THEN CASE WHEN `model_type` IN (2, 3) THEN 'openai' ELSE 'openai_compatible' END
  WHEN 'wan' THEN 'dashscope'
  WHEN 'wan_video' THEN 'dashscope'
  WHEN 'seedance' THEN 'volcengine'
  ELSE NULLIF(LOWER(`model_protocol`), '')
END;
