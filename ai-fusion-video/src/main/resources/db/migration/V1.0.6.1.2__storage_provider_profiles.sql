ALTER TABLE `afv_storage_config`
  ADD COLUMN `provider` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'S3兼容厂商：generic_s3/aliyun_oss/tencent_cos/qiniu_kodo/ctyun_zos/minio' AFTER `type`,
  ADD COLUMN `options` json NULL COMMENT '厂商扩展配置JSON' AFTER `custom_domain`;

UPDATE `afv_storage_config`
SET `provider` = CASE `type`
  WHEN 'aliyun_oss' THEN 'aliyun_oss'
  WHEN 'tencent_cos' THEN 'tencent_cos'
  WHEN 's3' THEN COALESCE(`provider`, 'generic_s3')
  ELSE `provider`
END
WHERE `type` IN ('aliyun_oss', 'tencent_cos', 's3');

UPDATE `afv_storage_config`
SET `type` = 's3'
WHERE `type` IN ('aliyun_oss', 'tencent_cos');
