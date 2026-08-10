ALTER TABLE `afv_ai_model`
  ADD COLUMN `model_family` VARCHAR(64) NULL DEFAULT NULL COMMENT '模型家族标识' AFTER `code`,
  ADD COLUMN `model_protocol` VARCHAR(64) NULL DEFAULT NULL COMMENT '模型协议标识' AFTER `model_family`;

UPDATE `afv_ai_model` m
LEFT JOIN `afv_api_config` c ON c.`id` = m.`api_config_id`
SET
  m.`model_family` = COALESCE(NULLIF(m.`model_family`, ''), CASE
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%jimeng%' OR COALESCE(m.`name`, '') LIKE '%即梦%' THEN 'jimeng'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%kling%' OR COALESCE(m.`name`, '') LIKE '%可灵%' THEN 'kling'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%sora%' THEN 'sora'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%seedance%' THEN 'seedance'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%wan%' OR COALESCE(m.`name`, '') LIKE '%万相%' THEN CASE WHEN m.`model_type` = 3 THEN 'wan_video' ELSE 'wan' END
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%veo%' THEN 'veo'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%pixverse%' THEN 'pixverse'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%hailuo%' OR COALESCE(m.`name`, '') LIKE '%海螺%' THEN 'hailuo'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%claude%' THEN 'claude'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%gemini%' THEN 'gemini'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%deepseek%' THEN 'deepseek'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%qwen%' OR COALESCE(m.`name`, '') LIKE '%千问%' THEN 'qwen'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%gpt%' OR LOWER(COALESCE(m.`code`, '')) REGEXP '^o[134]' THEN 'gpt'
    WHEN LOWER(COALESCE(c.`platform`, '')) = 'anthropic' THEN 'claude'
    WHEN LOWER(COALESCE(c.`platform`, '')) IN ('gemini', 'vertex_ai', 'vertexai') THEN 'gemini'
    WHEN LOWER(COALESCE(c.`platform`, '')) = 'dashscope' THEN CASE WHEN m.`model_type` = 3 THEN 'wan_video' ELSE 'qwen' END
    WHEN LOWER(COALESCE(c.`platform`, '')) = 'volcengine' THEN CASE WHEN m.`model_type` = 3 THEN 'seedance' ELSE 'doubao' END
    ELSE 'generic'
  END),
  m.`model_protocol` = COALESCE(NULLIF(m.`model_protocol`, ''), CASE
    WHEN m.`model_type` <> 3 THEN 'generic'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%jimeng%' OR COALESCE(m.`name`, '') LIKE '%即梦%' THEN 'jimeng'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%kling%' OR COALESCE(m.`name`, '') LIKE '%可灵%' THEN 'kling'
    WHEN LOWER(COALESCE(m.`code`, '')) LIKE '%sora%' THEN 'sora'
    WHEN LOWER(COALESCE(c.`platform`, '')) = 'dashscope' THEN 'wan'
    WHEN LOWER(COALESCE(c.`platform`, '')) = 'volcengine' THEN 'seedance'
    WHEN LOWER(COALESCE(c.`platform`, '')) = 'googleflowreverseapi' THEN 'google_flow'
    ELSE 'generic'
  END)
WHERE m.`model_family` IS NULL OR m.`model_family` = '' OR m.`model_protocol` IS NULL OR m.`model_protocol` = '';