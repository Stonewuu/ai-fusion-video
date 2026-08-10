ALTER TABLE afv_storyboard_episode
    ADD COLUMN composed_video_url VARCHAR(512) NULL COMMENT '本集合成视频URL',
    ADD COLUMN compose_status TINYINT NOT NULL DEFAULT 0 COMMENT '合成状态: 0未开始 1合成中 2已完成 3失败',
    ADD COLUMN compose_error_msg VARCHAR(1024) NULL COMMENT '合成失败原因',
    ADD COLUMN composed_at DATETIME NULL COMMENT '合成完成时间';
