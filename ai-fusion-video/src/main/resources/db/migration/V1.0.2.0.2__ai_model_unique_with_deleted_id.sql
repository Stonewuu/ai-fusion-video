ALTER TABLE `afv_ai_model`
  ADD COLUMN `deleted_id` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除隔离标识，0-未删除，删除后为记录ID' AFTER `deleted`;

UPDATE `afv_ai_model`
SET `deleted_id` = `id`
WHERE `deleted` = 1
  AND `deleted_id` = 0;

ALTER TABLE `afv_ai_model`
  DROP INDEX `uk_api_config_code`,
  ADD UNIQUE INDEX `uk_api_config_code`(`api_config_id` ASC, `code` ASC, `deleted_id` ASC) USING BTREE;