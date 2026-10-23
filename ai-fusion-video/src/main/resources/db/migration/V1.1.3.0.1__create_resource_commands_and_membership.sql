-- 为业务内容增加版本号，并建立命令、变更通知、内容历史和成员变更记录。

ALTER TABLE afv_project
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本';

ALTER TABLE afv_asset
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本';

ALTER TABLE afv_asset_item
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本';

ALTER TABLE afv_storyboard
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本';

ALTER TABLE afv_storyboard_episode
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本',
    ADD COLUMN compose_job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL COMMENT '最近一次媒体合成任务标识';

ALTER TABLE afv_storyboard_scene
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本';

ALTER TABLE afv_storyboard_item
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '业务内容版本';

CREATE TABLE `afv_resource_command` (
  `id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '业务命令标识',
  `actor_user_id` bigint NOT NULL COMMENT '经过认证的操作用户编号',
  `request_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '用户作用域内的幂等标识',
  `request_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '命令内容与执行作用域摘要',
  `origin` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '命令来源',
  `canvas_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '发起命令的画布标识',
  `agent_run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '可信智能体运行标识',
  `tool_call_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '可信工具调用标识',
  `request_json` json NOT NULL COMMENT '已验证的业务命令内容',
  `result_json` json DEFAULT NULL COMMENT '已提交的幂等回执',
  `undo_json` json DEFAULT NULL COMMENT '授权恢复所需的变更前值',
  `create_time` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '命令创建时间',
  `committed_at` datetime(6) DEFAULT NULL COMMENT '命令完成时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_resource_command_request` (`actor_user_id`,`request_id`),
  KEY `idx_resource_command_agent` (`agent_run_id`,`tool_call_id`),
  CONSTRAINT `ck_resource_command_actor` CHECK ((`actor_user_id` > 0)),
  CONSTRAINT `ck_resource_command_commit` CHECK ((((`result_json` is null) and (`committed_at` is null)) or ((`result_json` is not null) and (`committed_at` is not null)))),
  CONSTRAINT `ck_resource_command_json` CHECK (((json_type(`request_json`) = _utf8mb4'OBJECT') and ((`result_json` is null) or (json_type(`result_json`) = _utf8mb4'OBJECT')))),
  CONSTRAINT `ck_resource_command_origin` CHECK ((`origin` in (_utf8mb4'user',_utf8mb4'agent',_utf8mb4'system')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='跨页面与智能体共用的业务命令审计';

CREATE TABLE `afv_resource_event` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '资源事件顺序编号',
  `event_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '资源事件唯一标识',
  `command_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '所属业务命令标识',
  `resource_type` varchar(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '业务资源类型',
  `resource_id` bigint NOT NULL COMMENT '业务对象编号',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目编号',
  `resource_version` int NOT NULL COMMENT '提交后的业务版本',
  `deleted` tinyint(1) NOT NULL DEFAULT '0' COMMENT '对象是否已删除',
  `changed_fields` json NOT NULL COMMENT '本次变更的字段名称',
  `published` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已投递至相关画布',
  `create_time` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '事件提交时间',
  `published_at` datetime(6) DEFAULT NULL COMMENT '完成画布投递的时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_resource_event_id` (`event_id`),
  UNIQUE KEY `uk_resource_event_version` (`resource_type`,`resource_id`,`resource_version`),
  KEY `idx_resource_event_dispatch` (`published`,`id`),
  KEY `idx_resource_event_project` (`project_id`,`id`),
  KEY `fk_resource_event_command` (`command_id`),
  CONSTRAINT `fk_resource_event_command` FOREIGN KEY (`command_id`) REFERENCES `afv_resource_command` (`id`),
  CONSTRAINT `ck_resource_event_fields` CHECK ((json_type(`changed_fields`) = _utf8mb4'ARRAY')),
  CONSTRAINT `ck_resource_event_version` CHECK (((`resource_id` > 0) and (`resource_version` >= 0)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='业务资源变更日志与事务发件箱';

CREATE TABLE `afv_resource_event_canvas` (
  `event_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '已投递资源事件标识',
  `canvas_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '接收画布标识',
  `sequence_no` bigint NOT NULL COMMENT '画布内已提交的事件序号',
  `create_time` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '投递时间',
  PRIMARY KEY (`event_id`,`canvas_id`),
  KEY `fk_resource_projection_canvas` (`canvas_id`),
  CONSTRAINT `fk_resource_projection_canvas` FOREIGN KEY (`canvas_id`) REFERENCES `afv_canvas` (`id`),
  CONSTRAINT `fk_resource_projection_event` FOREIGN KEY (`event_id`) REFERENCES `afv_resource_event` (`event_id`),
  CONSTRAINT `ck_resource_projection_sequence` CHECK ((`sequence_no` > 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='资源事件到画布的幂等投递记录';

CREATE TABLE `afv_resource_feed_head` (
  `id` tinyint NOT NULL COMMENT '固定流标识',
  `last_sequence` bigint NOT NULL DEFAULT '0' COMMENT '最后提交的事件序号',
  PRIMARY KEY (`id`),
  CONSTRAINT `ck_resource_feed_head` CHECK (((`id` = 1) and (`last_sequence` >= 0)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='跨页面资源事件流提交序号';

CREATE TABLE `afv_resource_feed` (
  `sequence_no` bigint NOT NULL COMMENT '按投递事务提交顺序分配的序号',
  `event_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '资源事件唯一标识',
  `published_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '提交到全局资源流的时间',
  PRIMARY KEY (`sequence_no`),
  UNIQUE KEY `uk_resource_feed_event` (`event_id`),
  CONSTRAINT `fk_resource_feed_event` FOREIGN KEY (`event_id`) REFERENCES `afv_resource_event` (`event_id`),
  CONSTRAINT `ck_resource_feed_sequence` CHECK ((`sequence_no` > 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='原页面和画布共用的资源变更续读日志';

CREATE TABLE `afv_resource_revision` (
  `command_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '所属已提交业务命令',
  `resource_type` varchar(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '业务资源类型',
  `resource_id` bigint NOT NULL COMMENT '业务对象编号',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目编号',
  `resource_version` int NOT NULL COMMENT '本次命令提交后的对象版本',
  `before_json` json DEFAULT NULL COMMENT '修改前的完整业务内容，新增对象为空',
  `after_json` json NOT NULL COMMENT '修改后的完整业务内容，包含删除状态',
  `changed_fields` json NOT NULL COMMENT '本次实际修改的业务字段',
  `create_time` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '版本记录时间',
  PRIMARY KEY (`command_id`,`resource_type`,`resource_id`),
  KEY `idx_resource_revision_object` (`resource_type`,`resource_id`,`resource_version`),
  KEY `idx_resource_revision_project` (`project_id`,`create_time`,`command_id`),
  CONSTRAINT `fk_resource_revision_command` FOREIGN KEY (`command_id`) REFERENCES `afv_resource_command` (`id`),
  CONSTRAINT `ck_resource_revision_content` CHECK ((((`before_json` is null) or (json_type(`before_json`) = _utf8mb4'OBJECT')) and (json_type(`after_json`) = _utf8mb4'OBJECT') and (json_type(`changed_fields`) = _utf8mb4'ARRAY'))),
  CONSTRAINT `ck_resource_revision_version` CHECK (((`resource_id` > 0) and (`resource_version` >= 0)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='业务内容前后版本与条件恢复依据';

CREATE TABLE `afv_project_membership_state` (
  `project_id` bigint NOT NULL COMMENT '项目标识',
  `revision` bigint NOT NULL DEFAULT '0' COMMENT '成员权限版本',
  PRIMARY KEY (`project_id`),
  CONSTRAINT `fk_project_membership_state_project` FOREIGN KEY (`project_id`) REFERENCES `afv_project` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='项目成员权限版本';

CREATE TABLE `afv_project_membership_command` (
  `id` char(36) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '权限操作标识',
  `project_id` bigint NOT NULL COMMENT '项目标识',
  `revision` bigint NOT NULL COMMENT '提交后的成员权限版本',
  `actor_user_id` bigint NOT NULL COMMENT '操作用户标识',
  `actor_team_id` bigint DEFAULT NULL COMMENT '操作时团队标识',
  `request_id` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '幂等请求标识',
  `request_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '请求内容摘要',
  `user_id` bigint NOT NULL COMMENT '被调整的成员标识',
  `previous_role` int DEFAULT NULL COMMENT '调整前的单独项目角色',
  `current_role` int DEFAULT NULL COMMENT '调整后的单独项目角色，空值表示移除',
  `removed_canvas_grants` json NOT NULL COMMENT '随成员移除撤销的画布单独授权',
  `published` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已通知项目画布',
  `create_time` datetime(6) NOT NULL COMMENT '权限变更时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_project_membership_request` (`project_id`,`actor_user_id`,`request_id`),
  UNIQUE KEY `uk_project_membership_revision` (`project_id`,`revision`),
  KEY `idx_project_membership_outbox` (`published`,`create_time`),
  CONSTRAINT `fk_project_membership_command_project` FOREIGN KEY (`project_id`) REFERENCES `afv_project` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='项目成员权限审计与持久化通知';

INSERT INTO afv_resource_feed_head (id, last_sequence) VALUES (1, 0);
