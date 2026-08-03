ALTER TABLE `afv_ai_model`
    ADD COLUMN `reasoning_effort_levels` json NULL COMMENT '思考等级列表，按能力从高到低排序'
        AFTER `support_reasoning`;

UPDATE `afv_ai_model`
SET `reasoning_effort_levels` = JSON_ARRAY()
WHERE `reasoning_effort_levels` IS NULL;

ALTER TABLE `afv_ai_model`
    MODIFY COLUMN `reasoning_effort_levels` json NOT NULL COMMENT '思考等级列表，按能力从高到低排序';
