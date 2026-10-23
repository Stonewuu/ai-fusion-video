-- 固定生成请求和执行状态，支持媒体合成、预览及工作流试运行。

ALTER TABLE afv_image_item
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '生成图片条目的内容版本';

ALTER TABLE afv_video_item
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '生成视频条目的内容版本';

ALTER TABLE afv_image_task
    ADD COLUMN execution_context_json JSON NULL COMMENT '提交时验证并冻结的用户及智能体执行上下文';

ALTER TABLE afv_video_task
    ADD COLUMN execution_context_json JSON NULL COMMENT '提交时验证并冻结的用户及智能体执行上下文';

ALTER TABLE afv_comfyui_workflow_version
    ADD COLUMN last_test_run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL COMMENT '最近受理的持久化试运行',
    ADD COLUMN test_api_config_hash CHAR(64) NULL COMMENT '成功试运行对应的供应商连接摘要';

CREATE TABLE `afv_media_job` (
  `id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体任务标识',
  `kind` varchar(40) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '任务类型',
  `project_id` bigint DEFAULT NULL COMMENT '项目标识，独立任务可为空',
  `target_id` bigint DEFAULT NULL COMMENT '需要正式写回的业务对象标识',
  `target_version` int DEFAULT NULL COMMENT '等待写回的业务版本',
  `actor_user_id` bigint NOT NULL COMMENT '发起用户',
  `request_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '客户端幂等请求标识',
  `request_hash` char(64) CHARACTER SET ascii COLLATE ascii_general_ci NOT NULL COMMENT '请求与身份摘要',
  `context_json` json NOT NULL COMMENT '经认证的执行上下文',
  `input_json` json NOT NULL COMMENT '固定的输入来源版本与参数',
  `checkpoint_json` json NOT NULL COMMENT '已持久化的阶段结果',
  `status` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'QUEUED' COMMENT '任务状态',
  `phase` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '等待处理' COMMENT '当前处理阶段',
  `error_msg` text COLLATE utf8mb4_unicode_ci COMMENT '终止或待处理原因',
  `owner_instance_id` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '当前执行实例',
  `owner_epoch` bigint NOT NULL DEFAULT '0' COMMENT '执行代数',
  `lease_until` datetime(6) DEFAULT NULL COMMENT '执行租约截止时间',
  `next_run_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '下次执行时间',
  `deadline_at` datetime(6) NOT NULL COMMENT '任务总时限',
  `cancel_requested` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否请求取消',
  `revision` bigint NOT NULL DEFAULT '0' COMMENT '状态版本',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_media_job_request` (`actor_user_id`,`request_id`),
  KEY `idx_media_job_schedule` (`status`,`next_run_at`,`lease_until`),
  KEY `idx_media_job_target` (`project_id`,`target_id`,`created_at`),
  KEY `idx_media_job_kind_schedule` (`kind`,`status`,`next_run_at`,`lease_until`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='可恢复媒体处理任务';

CREATE TABLE `afv_media_job_submission` (
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体任务标识',
  `step` varchar(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '付费提交步骤',
  `attempt_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '唯一提交尝试',
  `provider_task_id` varchar(256) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '供应商确认的任务标识',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '提交意图登记时间',
  `acknowledged_at` datetime(6) DEFAULT NULL COMMENT '供应商回执登记时间',
  PRIMARY KEY (`job_id`,`step`),
  UNIQUE KEY `uk_media_job_submission_attempt` (`attempt_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='媒体付费提交意图与回执，未确认时禁止自动重发';

CREATE TABLE `afv_generation_submission` (
  `id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '受理记录标识',
  `request_id` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '调用方幂等标识',
  `scope_key` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户团队与入口作用域摘要',
  `request_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '规范化请求摘要',
  `kind` varchar(24) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '生成类别',
  `task_id` bigint NOT NULL COMMENT '业务任务数据库标识',
  `public_task_id` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '业务任务公开标识',
  `actor_user_id` bigint NOT NULL COMMENT '发起用户',
  `actor_team_id` bigint DEFAULT NULL COMMENT '发起团队上下文',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目',
  `canvas_node_run_id` char(36) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '画布节点运行标识',
  `context_json` json NOT NULL COMMENT '经校验的内部执行身份',
  `model_snapshot_json` json NOT NULL COMMENT '模型与能力的不可变快照',
  `api_config_id` bigint NOT NULL COMMENT '供应商配置标识',
  `api_config_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '供应商配置一致性摘要',
  `state` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '受理及聚合执行状态',
  `cancel_requested` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否请求停止后续执行',
  `revision` bigint NOT NULL DEFAULT '1' COMMENT '状态版本',
  `created_at` datetime(6) NOT NULL COMMENT '受理时间（UTC）',
  `updated_at` datetime(6) NOT NULL COMMENT '最近状态变化时间（UTC）',
  `agent_invocation_key` char(64) CHARACTER SET ascii COLLATE ascii_general_ci GENERATED ALWAYS AS ((case when (json_unquote(json_extract(`context_json`,_utf8mb4'$.origin')) = _utf8mb4'agent') then sha2(cast(json_array(json_unquote(json_extract(`context_json`,_utf8mb4'$.agentRunId')),json_unquote(json_extract(`context_json`,_utf8mb4'$.toolCallId'))) as char charset utf8mb4),256) else NULL end)) STORED COMMENT '智能体运行与工具调用的稳定身份摘要',
  `canvas_invocation_key` char(64) CHARACTER SET ascii COLLATE ascii_general_ci GENERATED ALWAYS AS ((case when (`canvas_node_run_id` is not null) then sha2(cast(json_array(`canvas_node_run_id`,coalesce(`agent_invocation_key`,_utf8mb4'single')) as char charset utf8mb4),256) else NULL end)) STORED COMMENT '画布执行与智能体工具调用的唯一受理身份',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_generation_request` (`scope_key`,`request_id`),
  UNIQUE KEY `uk_generation_task` (`kind`,`task_id`),
  UNIQUE KEY `uk_generation_agent_invocation` (`scope_key`,`agent_invocation_key`),
  UNIQUE KEY `uk_canvas_generation_invocation` (`canvas_invocation_key`),
  KEY `idx_generation_actor` (`actor_user_id`,`created_at`),
  KEY `idx_generation_project_state` (`project_id`,`state`),
  KEY `ix_generation_canvas_node` (`canvas_node_run_id`,`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='生成任务原子受理与幂等回执';

CREATE TABLE `afv_remote_generation_job` (
  `id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '远端执行记录标识',
  `submission_id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '所属受理记录',
  `item_id` bigint NOT NULL COMMENT '对应的结果条目标识',
  `item_index` int NOT NULL COMMENT '请求中的稳定结果顺序',
  `model_id` bigint NOT NULL COMMENT '占用额度的模型',
  `actor_user_id` bigint NOT NULL COMMENT '占用额度的用户',
  `project_id` bigint DEFAULT NULL COMMENT '占用额度的项目',
  `state` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '远端执行状态',
  `phase` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '当前持久化执行阶段',
  `provider_task_id` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '供应商受理标识',
  `active_attempt_id` char(36) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '当前显式提交尝试标识',
  `input_json` json NOT NULL COMMENT '固定生成参数与稳定媒体引用',
  `result_json` json DEFAULT NULL COMMENT '供应商结果接收检查点',
  `cancel_requested` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否请求远端取消',
  `cancel_sent` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已发送远端取消请求',
  `permit_held` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否仍占用远端并发额度',
  `owner_instance_id` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '当前工作者实例',
  `owner_epoch` bigint NOT NULL DEFAULT '0' COMMENT '工作者领取代数',
  `lease_until` datetime(6) DEFAULT NULL COMMENT '租约到期时间（UTC）',
  `next_poll_at` datetime(6) NOT NULL COMMENT '下次执行时间（UTC）',
  `deadline_at` datetime(6) NOT NULL COMMENT '自动等待截止时间（UTC）',
  `poll_attempts` int NOT NULL DEFAULT '0' COMMENT '单次查询累计次数',
  `consecutive_errors` int NOT NULL DEFAULT '0' COMMENT '连续传输错误次数',
  `error_code` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '稳定错误码',
  `error_message` text COLLATE utf8mb4_unicode_ci COMMENT '用户可读执行原因',
  `revision` bigint NOT NULL DEFAULT '1' COMMENT '状态版本',
  `created_at` datetime(6) NOT NULL COMMENT '创建时间（UTC）',
  `updated_at` datetime(6) NOT NULL COMMENT '最近变更时间（UTC）',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_remote_generation_item` (`submission_id`,`item_index`),
  KEY `idx_remote_generation_due` (`state`,`next_poll_at`,`lease_until`),
  KEY `idx_remote_generation_model` (`model_id`,`permit_held`),
  KEY `idx_remote_generation_actor` (`actor_user_id`,`permit_held`),
  KEY `idx_remote_generation_project` (`project_id`,`permit_held`),
  CONSTRAINT `fk_remote_generation_submission` FOREIGN KEY (`submission_id`) REFERENCES `afv_generation_submission` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='单次供应商请求与可恢复执行状态';

CREATE TABLE `afv_generation_attempt` (
  `id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '提交尝试标识',
  `job_id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '远端执行记录',
  `attempt_no` int NOT NULL COMMENT '经明确决策创建的尝试序号',
  `request_json` json DEFAULT NULL COMMENT '可查询协议的预备请求检查点',
  `provider_task_id` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '不可变供应商受理回执',
  `result_json` json DEFAULT NULL COMMENT '同步供应商的不可变响应结果',
  `resolution` varchar(32) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '未知受理的核实结论',
  `resolution_note` text COLLATE utf8mb4_unicode_ci COMMENT '核实依据与明确重试说明',
  `resolved_by` bigint DEFAULT NULL COMMENT '作出核实决定的用户',
  `created_at` datetime(6) NOT NULL COMMENT '调用前提交意图时间（UTC）',
  `submitted_at` datetime(6) DEFAULT NULL COMMENT '准备完成后、付费调用之前提交的意图时间（UTC）',
  `acknowledged_at` datetime(6) DEFAULT NULL COMMENT '回执入库时间（UTC）',
  `resolved_at` datetime(6) DEFAULT NULL COMMENT '核实决定时间（UTC）',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_generation_attempt_no` (`job_id`,`attempt_no`),
  CONSTRAINT `fk_generation_attempt_job` FOREIGN KEY (`job_id`) REFERENCES `afv_remote_generation_job` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='外部副作用意图、回执与人工核实审计';

CREATE TABLE `afv_async_outbox` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '投递记录标识',
  `event_key` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '事件幂等标识',
  `event_type` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '已登记投递类型',
  `aggregate_id` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '关联聚合标识',
  `payload_json` json NOT NULL COMMENT '不含凭据的投递内容',
  `state` varchar(16) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'PENDING' COMMENT '待投递或已投递',
  `attempts` int NOT NULL DEFAULT '0' COMMENT '投递尝试次数',
  `next_attempt_at` datetime(6) NOT NULL COMMENT '下次投递时间（UTC）',
  `last_error` text COLLATE utf8mb4_unicode_ci COMMENT '最近投递错误',
  `created_at` datetime(6) NOT NULL COMMENT '与业务状态一起提交的创建时间（UTC）',
  `delivered_at` datetime(6) DEFAULT NULL COMMENT '投递确认时间（UTC）',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_async_outbox_event` (`event_key`),
  KEY `idx_async_outbox_due` (`state`,`next_attempt_at`,`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='业务任务可靠投递发件箱';

CREATE TABLE `afv_generation_event` (
  `sequence` bigint NOT NULL AUTO_INCREMENT COMMENT '按投递提交顺序分配的游标',
  `outbox_id` bigint NOT NULL COMMENT '来源发件箱记录',
  `submission_id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '关联受理记录',
  `revision` bigint NOT NULL COMMENT '聚合状态版本',
  `event_type` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '状态变化类型',
  `payload_json` json NOT NULL COMMENT '用户可见状态概要',
  `created_at` datetime(6) NOT NULL COMMENT '事件提交时间（UTC）',
  PRIMARY KEY (`sequence`),
  UNIQUE KEY `uk_generation_event_outbox` (`outbox_id`),
  KEY `idx_generation_event_submission` (`submission_id`,`sequence`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='可续读的生成状态事件';

CREATE TABLE `afv_generation_dispatch_lock` (
  `id` int NOT NULL COMMENT '全局额度检查锁标识',
  `version` bigint NOT NULL DEFAULT '0' COMMENT '调度版本',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='跨实例生成额度与事件顺序的短事务锁';

CREATE TABLE `afv_generation_request_withdrawal` (
  `scope_key` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户团队与入口作用域摘要',
  `request_id` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '调用方原始幂等请求标识',
  `kind` varchar(24) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '生成类别',
  `actor_user_id` bigint NOT NULL COMMENT '撤回用户',
  `actor_team_id` bigint DEFAULT NULL COMMENT '撤回时的团队上下文',
  `created_at` datetime(6) NOT NULL COMMENT '撤回时间（UTC）',
  PRIMARY KEY (`scope_key`,`request_id`),
  KEY `idx_generation_withdrawal_actor` (`actor_user_id`,`created_at`),
  CONSTRAINT `ck_generation_withdrawal_kind` CHECK ((`kind` in (_utf8mb4'IMAGE',_utf8mb4'VIDEO')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='生成提交撤回凭据，阻止迟到请求重新创建付费任务';

CREATE TABLE `afv_comfyui_test_run` (
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '共享媒体任务标识',
  `workflow_id` bigint NOT NULL COMMENT '工作流标识',
  `version_id` bigint NOT NULL COMMENT '试运行的工作流版本',
  `workflow_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '受理时的工作流内容摘要',
  `api_config_id` bigint NOT NULL COMMENT '目标供应商配置',
  `api_config_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '供应商连接一致性摘要',
  `actor_team_id` bigint DEFAULT NULL COMMENT '受理时的团队作用域',
  `prompt_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '提交前固定的远端请求标识',
  `request_json` json DEFAULT NULL COMMENT '已准备的远端请求，禁止直接暴露给客户端',
  `submitted_at` datetime(6) DEFAULT NULL COMMENT '实际提交意图时间（UTC）',
  `acknowledged_at` datetime(6) DEFAULT NULL COMMENT '不可变受理回执时间（UTC）',
  `result_json` json DEFAULT NULL COMMENT '已确认的远端结果检查点',
  `permit_held` tinyint(1) NOT NULL DEFAULT '0' COMMENT '远端任务是否仍占用试运行额度',
  `cancel_sent` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已经发送远端取消请求',
  `poll_attempts` int NOT NULL DEFAULT '0' COMMENT '单次查询次数',
  `consecutive_errors` int NOT NULL DEFAULT '0' COMMENT '连续查询失败次数',
  PRIMARY KEY (`job_id`),
  UNIQUE KEY `uk_comfyui_test_prompt` (`prompt_id`),
  KEY `idx_comfyui_test_version` (`version_id`,`job_id`),
  KEY `idx_comfyui_test_permit` (`api_config_id`,`permit_held`),
  CONSTRAINT `fk_comfyui_test_media_job` FOREIGN KEY (`job_id`) REFERENCES `afv_media_job` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='ComfyUI 管理试运行的提交回执与调度检查点';

CREATE TABLE `afv_media_job_event` (
  `sequence` bigint NOT NULL AUTO_INCREMENT COMMENT '按发布顺序分配的事件游标',
  `outbox_id` bigint NOT NULL COMMENT '对应的可靠发件箱记录',
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体任务标识',
  `revision` bigint NOT NULL COMMENT '任务状态版本',
  `event_type` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '已登记的事件类型',
  `payload_json` json NOT NULL COMMENT '不包含凭据的状态或核实记录',
  `created_at` datetime(6) NOT NULL COMMENT '事件发布时间（UTC）',
  PRIMARY KEY (`sequence`),
  UNIQUE KEY `uk_media_job_event_outbox` (`outbox_id`),
  KEY `idx_media_job_event_job` (`job_id`,`sequence`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='媒体任务状态与人工核实的可追溯事件';

CREATE TABLE `afv_comfyui_test_withdrawal` (
  `actor_user_id` bigint NOT NULL COMMENT '请求发起用户',
  `request_key` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '被撤回的试运行幂等标识',
  `actor_team_id` bigint DEFAULT NULL COMMENT '撤回时核验的团队作用域',
  `created_at` datetime(6) NOT NULL COMMENT '撤回时间（UTC）',
  PRIMARY KEY (`actor_user_id`,`request_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='阻止试运行撤回后迟到请求再次受理';

CREATE TABLE `afv_episode_audio_job` (
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '分集合成任务标识',
  `api_config_id` bigint NOT NULL COMMENT '冻结的音频服务配置编号',
  `api_config_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '音频服务连接摘要',
  `sfx_enabled` tinyint(1) NOT NULL COMMENT '是否需要生成音效',
  `permit_held` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否预留或持有远端音频并发额度',
  `music_result_json` json DEFAULT NULL COMMENT '已确认的配乐终态响应，禁止直接暴露给客户端',
  `music_result_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '不可变配乐响应摘要',
  `sfx_result_json` json DEFAULT NULL COMMENT '已确认的音效终态响应，禁止直接暴露给客户端',
  `sfx_result_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '不可变音效响应摘要',
  `consecutive_errors` int NOT NULL DEFAULT '0' COMMENT '连续查询失败次数',
  `resolved_steps_json` json NOT NULL DEFAULT (json_array()) COMMENT '已人工确认远端结束的步骤，保留原提交和响应',
  `review_json` json DEFAULT NULL COMMENT '最近一次人工核实的依据、原身份和动作摘要',
  `legacy_state` varchar(32) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '升级前缺少终态证据的本地任务状态',
  PRIMARY KEY (`job_id`),
  KEY `idx_episode_audio_capacity` (`api_config_id`,`permit_held`),
  CONSTRAINT `fk_episode_audio_job` FOREIGN KEY (`job_id`) REFERENCES `afv_media_job` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='分集配乐和音效的并发占用与不可变远端响应';

CREATE TABLE `afv_execution_recovery_decision` (
  `id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '管理员核实决定标识',
  `admin_user_id` bigint NOT NULL COMMENT '实际操作的管理员编号',
  `request_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '管理员请求幂等标识',
  `request_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '具体核实决定的不可变摘要',
  `execution_kind` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '被核实的执行类型',
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '被核实的持久化任务标识',
  `expected_revision` bigint NOT NULL COMMENT '管理员看到并确认的任务版本',
  `actor_user_id` bigint NOT NULL COMMENT '原请求发起用户，禁止改写为管理员',
  `actor_team_id` bigint DEFAULT NULL COMMENT '原请求发起团队作用域',
  `project_id` bigint DEFAULT NULL COMMENT '原任务所属项目',
  `canvas_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '原任务所属画布',
  `facts_json` json NOT NULL COMMENT '核实前的脱敏状态与远端标识，不含提示词、结果内容或密钥',
  `note` varchar(2000) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '管理员核实远端已结束的依据',
  `receipt_json` json NOT NULL COMMENT '同一事务提交的任务收尾回执',
  `created_at` datetime(6) NOT NULL COMMENT '核实完成时间（UTC）',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_execution_recovery_request` (`admin_user_id`,`request_id`),
  KEY `idx_execution_recovery_job` (`execution_kind`,`job_id`,`created_at`),
  KEY `idx_execution_recovery_created` (`created_at`,`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='撤权执行的管理员核实与收尾审计，不授予业务写入权限';

CREATE TABLE `afv_media_preview_job` (
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '预览生成任务标识',
  `source_resource_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '原始媒体资源标识',
  `source_version` int NOT NULL COMMENT '固定的原始媒体版本',
  `profile` varchar(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '预览规格及算法版本',
  `sample_at_ms` bigint NOT NULL COMMENT '视频抽样位置，图片固定为零',
  `previous_job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '显式重试的上一任务标识',
  `output_resource_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '受原始媒体权限约束的预览资源',
  PRIMARY KEY (`job_id`),
  UNIQUE KEY `uk_media_preview_output` (`output_resource_id`),
  KEY `idx_media_preview_source` (`source_resource_id`,`source_version`,`profile`),
  KEY `fk_media_preview_previous` (`previous_job_id`),
  CONSTRAINT `fk_media_preview_job` FOREIGN KEY (`job_id`) REFERENCES `afv_media_job` (`id`),
  CONSTRAINT `fk_media_preview_output` FOREIGN KEY (`output_resource_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `fk_media_preview_previous` FOREIGN KEY (`previous_job_id`) REFERENCES `afv_media_job` (`id`),
  CONSTRAINT `fk_media_preview_source` FOREIGN KEY (`source_resource_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `ck_media_preview_distinct_output` CHECK (((`output_resource_id` is null) or (`output_resource_id` <> `source_resource_id`))),
  CONSTRAINT `ck_media_preview_sample` CHECK ((`sample_at_ms` >= 0)),
  CONSTRAINT `ck_media_preview_source_version` CHECK ((`source_version` >= 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='私有媒体预览任务与原始资源的不可变关联';

CREATE TABLE `afv_media_preview_current` (
  `source_resource_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '原始媒体资源标识',
  `source_version` int NOT NULL COMMENT '固定的原始媒体版本',
  `profile` varchar(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '预览规格及算法版本',
  `job_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '当前生效的预览生成任务',
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '当前任务切换时间',
  PRIMARY KEY (`source_resource_id`,`source_version`,`profile`),
  UNIQUE KEY `uk_media_preview_current_job` (`job_id`),
  CONSTRAINT `fk_media_preview_current_job` FOREIGN KEY (`job_id`) REFERENCES `afv_media_preview_job` (`job_id`),
  CONSTRAINT `fk_media_preview_current_source` FOREIGN KEY (`source_resource_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `ck_media_preview_current_version` CHECK ((`source_version` >= 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='按原媒体版本和规格去重的当前预览任务';

-- 历史个人任务已有明确发起用户；只补齐该已知个人归属，不从当前团队或项目猜测归属。
UPDATE afv_image_task SET owner_id = user_id WHERE owner_type = 1 AND owner_id IS NULL AND user_id > 0;

UPDATE afv_video_task SET owner_id = user_id WHERE owner_type = 1 AND owner_id IS NULL AND user_id > 0;

INSERT INTO afv_generation_dispatch_lock(id) VALUES (1);
