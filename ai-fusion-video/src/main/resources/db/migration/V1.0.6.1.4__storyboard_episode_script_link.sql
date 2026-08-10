ALTER TABLE `afv_storyboard_episode`
  ADD COLUMN `script_episode_id` bigint NULL DEFAULT NULL COMMENT '关联的剧本分集ID' AFTER `storyboard_id`,
  ADD COLUMN `deleted_id` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除隔离标识，0-未删除，删除后为记录ID' AFTER `deleted`;

UPDATE `afv_storyboard_episode`
SET `deleted_id` = `id`
WHERE `deleted` = 1
  AND `deleted_id` = 0;

ALTER TABLE `afv_storyboard_episode`
  ADD INDEX `idx_sb_episode_script_episode`(`script_episode_id` ASC) USING BTREE,
  ADD UNIQUE INDEX `uk_sb_episode_script_episode`(`storyboard_id` ASC, `script_episode_id` ASC, `deleted_id` ASC) USING BTREE;
