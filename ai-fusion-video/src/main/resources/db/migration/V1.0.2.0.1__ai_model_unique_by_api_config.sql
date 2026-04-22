ALTER TABLE `afv_ai_model`
  DROP INDEX `code`,
  ADD UNIQUE INDEX `uk_api_config_code`(`api_config_id` ASC, `code` ASC) USING BTREE;