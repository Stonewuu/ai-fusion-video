ALTER TABLE afv_storyboard_episode
    ADD COLUMN bgm_audio_url VARCHAR(512) NULL COMMENT '配乐音轨URL（持久化存档）',
    ADD COLUMN bgm_license_id VARCHAR(128) NULL COMMENT '配乐音轨授权凭证ID（商用留档）',
    ADD COLUMN bgm_sfx_audio_url VARCHAR(512) NULL COMMENT '音效音轨URL（持久化存档）',
    ADD COLUMN bgm_sfx_license_id VARCHAR(128) NULL COMMENT '音效音轨授权凭证ID（商用留档）',
    ADD COLUMN bgm_error_msg VARCHAR(1024) NULL COMMENT '配乐失败原因（配乐失败不影响本集合成结果）';
