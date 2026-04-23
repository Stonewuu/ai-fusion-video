-- 将 afv_api_config.app_secret 由 VARCHAR(512) 扩展为 TEXT
-- Data truncation: Data too long for column 'app_secret' 错误。

ALTER TABLE `afv_api_config`
  MODIFY COLUMN `app_secret` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '应用密钥/服务账号 JSON Key（部分平台需要，如 Vertex AI Service Account）';
