ALTER TABLE `afv_api_config`
  ADD COLUMN `proxy_type` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '出站代理类型：none/http/socks5' AFTER `auto_append_v1_path`,
  ADD COLUMN `proxy_host` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '出站代理主机' AFTER `proxy_type`,
  ADD COLUMN `proxy_port` int NULL DEFAULT NULL COMMENT '出站代理端口' AFTER `proxy_host`;