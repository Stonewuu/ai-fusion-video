ALTER TABLE afv_agent_message
    MODIFY COLUMN reasoning_content LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI推理过程内容（思维链）',
    MODIFY COLUMN content LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '消息正文、工具调用参数或工具结果';

ALTER TABLE afv_agent_event
    MODIFY COLUMN payload_json LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '智能体运行事件数据';
