ALTER TABLE `afv_api_config`
  ADD COLUMN `auto_append_v1_path` tinyint NOT NULL DEFAULT 1 COMMENT 'OpenAI兼容请求是否自动补充/v1路径' AFTER `api_url`;

UPDATE `afv_api_config`
SET `auto_append_v1_path` = 1
WHERE `auto_append_v1_path` IS NULL;