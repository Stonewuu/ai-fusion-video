-- 保存智能体执行作用域、用户配置和提供商身份。

ALTER TABLE afv_agent_run
    ADD COLUMN actor_team_id BIGINT NULL COMMENT '启动运行时经过认证的团队作用域',
    ADD CONSTRAINT ck_agent_run_actor_team CHECK (actor_team_id IS NULL OR actor_team_id > 0),
    ADD COLUMN canvas_scope_json JSON NULL COMMENT '受理时验证并固定的画布、选中节点及视野上下文' AFTER actor_team_id,
    ADD COLUMN external_check_after DATETIME(3) NULL COMMENT '外部任务下一次检查时间' AFTER canvas_scope_json,
    ADD COLUMN external_resume_json JSON NULL COMMENT '等待结果已提交但尚未交付执行器的恢复消息' AFTER external_check_after,
    ADD COLUMN external_resume_sequence BIGINT NULL COMMENT '恢复结果提交时的事件序号' AFTER external_resume_json,
    ADD COLUMN external_resume_at DATETIME(3) NULL COMMENT '恢复结果提交时间，用于排除已开始的模型调用' AFTER external_resume_sequence,
    ADD INDEX idx_agent_external_check (status, waiting_tool_name, external_check_after, id);

ALTER TABLE afv_agent_workspace_entry
    ADD COLUMN media_references_json JSON NULL COMMENT '正文中明确的托管媒体引用，空值表示历史正文尚待索引',
    ADD CONSTRAINT ck_workspace_media_references CHECK (media_references_json IS NULL OR JSON_TYPE(media_references_json)='ARRAY');

ALTER TABLE afv_api_config
    ADD COLUMN provider_id VARCHAR(64) NULL COMMENT '内置提供商标识，空值表示自定义接入',
    ADD COLUMN provider_setup_key CHAR(64) NULL COMMENT '快速配置请求的用户隔离幂等标识',
    ADD COLUMN provider_setup_hash CHAR(64) NULL COMMENT '首次快速配置请求摘要',
    ADD UNIQUE KEY uk_api_config_provider_setup (provider_setup_key);

CREATE TABLE `afv_agent_model_request` (
  `id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '模型调用标识，同时作为用量账本调用标识',
  `run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '实际执行智能体任务标识',
  `model_id` bigint NOT NULL COMMENT '冻结的模型编号',
  `request_owner` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '发起原调用的工作者标识',
  `request_epoch` bigint NOT NULL COMMENT '发起原调用的租约代次',
  `status` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型调用状态',
  `permit_held` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否仍占用模型并发额度',
  `response_received_at` datetime(6) DEFAULT NULL COMMENT '原调用完整响应接收时间（UTC）',
  `revision` bigint NOT NULL DEFAULT '1' COMMENT '核实版本',
  `created_at` datetime(6) NOT NULL COMMENT '发送意图落库时间（UTC）',
  `updated_at` datetime(6) NOT NULL COMMENT '最近变更时间（UTC）',
  PRIMARY KEY (`id`),
  KEY `idx_agent_model_capacity` (`model_id`,`permit_held`),
  KEY `idx_agent_model_request_run` (`run_id`,`status`),
  KEY `idx_agent_model_request_review` (`status`,`created_at`),
  CONSTRAINT `chk_agent_model_request_status` CHECK ((`status` in (_utf8mb4'RUNNING',_utf8mb4'COMPLETED',_utf8mb4'UNKNOWN_SUBMISSION',_utf8mb4'CANCELLED')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体模型请求与并发占用，未知远端状态禁止自动重发';

CREATE TABLE `afv_agent_user_override` (
  `user_id` bigint NOT NULL COMMENT '所属用户',
  `agent_type` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '内置智能体标识',
  `overrides_json` json NOT NULL COMMENT '用户覆盖的配置字段，缺省字段继承内置值',
  `revision` bigint NOT NULL DEFAULT '1' COMMENT '配置修订号，用于并发编辑校验',
  `created_at` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `updated_at` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
  PRIMARY KEY (`user_id`,`agent_type`),
  CONSTRAINT `chk_agent_user_override_revision` CHECK ((`revision` > 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户智能体覆盖配置';
