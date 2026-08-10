INSERT IGNORE INTO `afv_system_config` (`config_key`, `config_value`, `remark`, `deleted`)
SELECT
    'resource_base_url',
    `config_value`,
    '后端资源公网地址（由原站点公网地址拆分）',
    0
FROM `afv_system_config`
WHERE `config_key` = 'site_base_url'
  AND `deleted` = 0
  AND TRIM(COALESCE(`config_value`, '')) <> '';
