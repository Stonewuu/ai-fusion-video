ALTER TABLE `afv_api_config`
  ADD COLUMN `proxy_username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '出站代理认证用户名' AFTER `proxy_port`,
  ADD COLUMN `proxy_password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '出站代理认证密码' AFTER `proxy_username`;