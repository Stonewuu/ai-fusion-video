-- 将长文本字段由 TEXT 扩展为 MEDIUMTEXT
-- 解决长剧本解析与工具调用结果写入时超过 TEXT 上限导致的 Data truncation 错误。

ALTER TABLE `afv_script`
  MODIFY COLUMN `raw_content` MEDIUMTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '剧本原始内容（用户粘贴的原文）';

ALTER TABLE `afv_script_episode`
  MODIFY COLUMN `raw_content` MEDIUMTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '本集原始剧本内容';

ALTER TABLE `afv_agent_message`
  MODIFY COLUMN `content` MEDIUMTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '消息文本内容';