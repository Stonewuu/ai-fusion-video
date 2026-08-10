ALTER TABLE `afv_image_item`
  MODIFY COLUMN `platform_task_id` varchar(512)
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    NULL DEFAULT NULL COMMENT '平台侧任务ID';

ALTER TABLE `afv_video_item`
  MODIFY COLUMN `platform_task_id` varchar(512)
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    NULL DEFAULT NULL COMMENT '平台侧任务ID';