ALTER TABLE `afv_storyboard_item`
  ADD COLUMN `first_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '首帧参考图片URL' AFTER `generated_image_url`,
  ADD COLUMN `last_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '尾帧参考图片URL' AFTER `first_frame_image_url`,
  ADD COLUMN `first_frame_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI生成首帧时使用的提示词' AFTER `last_frame_image_url`,
  ADD COLUMN `last_frame_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI生成尾帧时使用的提示词' AFTER `first_frame_prompt`;

UPDATE `afv_storyboard_item`
SET `first_frame_image_url` = COALESCE(NULLIF(`generated_image_url`, ''), NULLIF(`image_url`, ''), NULLIF(`reference_image_url`, ''))
WHERE (`first_frame_image_url` IS NULL OR `first_frame_image_url` = '')
  AND COALESCE(NULLIF(`generated_image_url`, ''), NULLIF(`image_url`, ''), NULLIF(`reference_image_url`, '')) IS NOT NULL;
