-- 登记托管媒体、不可变文件、业务引用及清理状态。

CREATE TABLE `afv_media_resource` (
  `id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '不可变媒体资源标识',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目编号，独立素材为空',
  `owner_type` tinyint NOT NULL COMMENT '所有者类型：一为用户，二为团队',
  `owner_id` bigint NOT NULL COMMENT '所有者编号',
  `created_by` bigint NOT NULL COMMENT '经过认证的创建用户编号',
  `actor_team_id` bigint DEFAULT NULL COMMENT '创建时固定的团队作用域',
  `request_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '创建用户作用域内的幂等键',
  `request_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '内容及来源作用域摘要',
  `media_kind` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体类别',
  `mime_type` varchar(96) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '经内容检测的媒体类型',
  `original_name` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '安全处理后的原始文件名',
  `byte_size` bigint NOT NULL COMMENT '明文内容字节数',
  `content_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '明文内容校验摘要',
  `width` int DEFAULT NULL COMMENT '画面宽度',
  `height` int DEFAULT NULL COMMENT '画面高度',
  `duration_ms` bigint DEFAULT NULL COMMENT '持续时间毫秒数',
  `has_audio` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否包含音轨',
  `state` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '登记状态',
  `version` int NOT NULL DEFAULT '0' COMMENT '资源元数据版本',
  `blob_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '已完成或正在接收的密文对象标识',
  `ingest_epoch` bigint NOT NULL DEFAULT '1' COMMENT '接收者隔离代数',
  `lease_until` datetime(6) DEFAULT NULL COMMENT '本轮接收租约到期时间',
  `last_error` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '可向用户显示的接收失败原因',
  `origin` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '可信操作来源',
  `agent_run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '发起操作的智能体运行标识',
  `tool_call_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '发起操作的工具调用标识',
  `canvas_id` char(36) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '发起操作的画布标识',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '最近状态变更时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_media_request` (`created_by`,`request_id`),
  KEY `idx_media_project` (`project_id`,`created_at`,`id`),
  KEY `idx_media_owner` (`owner_type`,`owner_id`,`created_at`,`id`),
  KEY `idx_media_ingest` (`state`,`lease_until`),
  KEY `idx_media_resource_blob` (`blob_id`,`state`),
  CONSTRAINT `ck_media_dimensions` CHECK ((((`width` is null) and (`height` is null)) or ((`width` > 0) and (`height` > 0)))),
  CONSTRAINT `ck_media_duration` CHECK (((`duration_ms` is null) or (`duration_ms` > 0))),
  CONSTRAINT `ck_media_kind` CHECK ((`media_kind` in (_ascii'IMAGE',_ascii'VIDEO',_ascii'AUDIO',_ascii'TEXT',_ascii'DOCUMENT'))),
  CONSTRAINT `ck_media_owner` CHECK (((`owner_type` in (1,2)) and (`owner_id` > 0) and (`created_by` > 0))),
  CONSTRAINT `ck_media_size` CHECK (((`byte_size` > 0) and (`version` >= 0) and (`ingest_epoch` > 0))),
  CONSTRAINT `ck_media_state` CHECK ((`state` in (_ascii'STAGING',_ascii'READY',_ascii'FAILED',_ascii'DELETED')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='经授权登记的不可变媒体资源';

CREATE TABLE `afv_media_blob` (
  `id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '每轮接收独立的密文对象标识',
  `resource_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '对应媒体资源标识',
  `ingest_epoch` bigint NOT NULL COMMENT '所属接收代数',
  `storage_config_id` bigint NOT NULL COMMENT '实际使用的存储配置编号',
  `backend` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '已固定的存储后端',
  `object_key` varchar(512) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '密文对象相对位置，不可作为明文地址',
  `keyset_bytes` blob NOT NULL COMMENT '仅服务端可读的流式密钥集，禁止接口返回、缓存或日志输出',
  `stored_bytes` bigint DEFAULT NULL COMMENT '写入完成后的密文字节数',
  `state` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '对象接收与保留状态',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '预留时间',
  `orphaned_at` datetime(6) DEFAULT NULL COMMENT '确认接收失败或被新接收代数替代的时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_media_blob_attempt` (`resource_id`,`ingest_epoch`),
  KEY `idx_media_blob_storage` (`storage_config_id`,`state`),
  KEY `idx_media_blob_orphan` (`state`,`orphaned_at`,`id`),
  CONSTRAINT `fk_media_blob_resource` FOREIGN KEY (`resource_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `ck_media_blob_backend` CHECK ((`backend` in (_ascii'local',_ascii's3'))),
  CONSTRAINT `ck_media_blob_state` CHECK ((`state` in (_ascii'STAGING',_ascii'READY',_ascii'ORPHAN',_ascii'PURGED')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='托管媒体私有密文与服务器密钥，禁止向客户端投影';

CREATE TABLE `afv_media_source` (
  `resource_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体资源标识',
  `source_type` varchar(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '经过授权的来源业务类型',
  `source_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '来源业务对象标识',
  `source_version` int NOT NULL COMMENT '登记时固定的来源版本',
  `source_field` varchar(96) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '显式映射的来源字段',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '来源登记时间',
  PRIMARY KEY (`source_type`,`source_id`,`source_version`,`source_field`),
  KEY `idx_media_source_resource` (`resource_id`),
  CONSTRAINT `fk_media_source_resource` FOREIGN KEY (`resource_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `ck_media_source_version` CHECK ((`source_version` >= 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='业务来源到托管媒体的明确映射';

CREATE TABLE `afv_media_event` (
  `sequence_no` bigint NOT NULL AUTO_INCREMENT COMMENT '媒体事件顺序编号',
  `event_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体事件唯一标识',
  `resource_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体资源标识',
  `resource_version` int NOT NULL COMMENT '提交后的资源版本',
  `event_type` varchar(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '媒体事件类型',
  `actor_user_id` bigint NOT NULL COMMENT '经过认证的操作用户编号',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目编号',
  `payload_json` json NOT NULL COMMENT '不含密钥、凭据和存储配置的事件正文',
  `published` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否已向相关画布投递',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '事件提交时间',
  PRIMARY KEY (`sequence_no`),
  UNIQUE KEY `uk_media_event_id` (`event_id`),
  UNIQUE KEY `uk_media_event_version` (`resource_id`,`resource_version`),
  KEY `idx_media_event_dispatch` (`published`,`sequence_no`),
  CONSTRAINT `fk_media_event_resource` FOREIGN KEY (`resource_id`) REFERENCES `afv_media_resource` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='媒体状态审计与事务发件箱';

CREATE TABLE `afv_media_binding` (
  `resource_type` varchar(48) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '引用媒体的业务对象类型',
  `resource_id` bigint NOT NULL COMMENT '引用媒体的业务对象编号',
  `field_name` varchar(96) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '明确登记的媒体字段路径',
  `media_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '已就绪的托管媒体标识',
  `resource_version` int NOT NULL COMMENT '建立此引用时的业务版本',
  `bound_by` bigint NOT NULL COMMENT '经过认证的绑定操作用户编号',
  `bound_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '绑定提交时间',
  PRIMARY KEY (`resource_type`,`resource_id`,`field_name`),
  KEY `idx_media_binding_media` (`media_id`),
  CONSTRAINT `fk_media_binding_media` FOREIGN KEY (`media_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `ck_media_binding_resource` CHECK (((`resource_id` > 0) and (`resource_version` >= 0) and (`bound_by` > 0)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='业务字段到托管媒体的授权引用';

CREATE TABLE `afv_media_preset_binding` (
  `preset_key` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '内置画风预设标识',
  `media_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '已登记的内置参考图媒体标识',
  `published_by` bigint NOT NULL COMMENT '执行全局预设登记的管理员',
  `published_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '预设登记时间',
  PRIMARY KEY (`preset_key`),
  KEY `idx_media_preset_resource` (`media_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='面向已登录用户的内置画风媒体引用';

CREATE TABLE `afv_media_copy` (
  `media_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '具有独立归属的副本媒体标识',
  `source_media_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '经过授权复制的来源媒体标识',
  `source_version` int NOT NULL COMMENT '复制时的来源版本',
  `created_by` bigint NOT NULL COMMENT '发起复制的用户标识',
  `created_at` datetime(6) NOT NULL COMMENT '复制时间',
  PRIMARY KEY (`media_id`),
  KEY `idx_media_copy_source` (`source_media_id`),
  CONSTRAINT `fk_media_copy_source` FOREIGN KEY (`source_media_id`) REFERENCES `afv_media_resource` (`id`),
  CONSTRAINT `fk_media_copy_target` FOREIGN KEY (`media_id`) REFERENCES `afv_media_resource` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='共享不可变密文且权限独立的媒体副本来源';

CREATE TABLE `afv_media_blob_cleanup` (
  `blob_id` char(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '待清理的独立密文对象标识',
  `status` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '等待、保留引用阻止、已领取或已清理',
  `eligible_at` datetime(6) NOT NULL COMMENT '最早允许物理清理的时间',
  `next_attempt_at` datetime(6) NOT NULL COMMENT '下次检查或重试时间',
  `owner_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL COMMENT '当前清理工作者标识',
  `epoch` bigint NOT NULL DEFAULT '0' COMMENT '清理领取代数，隔离失效工作者',
  `lease_until` datetime(6) DEFAULT NULL COMMENT '领取租约截止时间',
  `attempts` int NOT NULL DEFAULT '0' COMMENT '实际领取次数',
  `last_error` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '不含密钥和存储路径的错误说明',
  `purged_at` datetime(6) DEFAULT NULL COMMENT '物理文件确认删除的时间',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '首次登记时间',
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '最近状态变更时间',
  PRIMARY KEY (`blob_id`),
  KEY `idx_media_cleanup_dispatch` (`status`,`next_attempt_at`,`lease_until`),
  CONSTRAINT `fk_media_cleanup_blob` FOREIGN KEY (`blob_id`) REFERENCES `afv_media_blob` (`id`),
  CONSTRAINT `ck_media_cleanup_counts` CHECK (((`epoch` >= 0) and (`attempts` >= 0))),
  CONSTRAINT `ck_media_cleanup_lease` CHECK ((((`status` = _utf8mb4'CLAIMED') and (`owner_id` is not null) and (`lease_until` is not null)) or ((`status` <> _utf8mb4'CLAIMED') and (`owner_id` is null) and (`lease_until` is null)))),
  CONSTRAINT `ck_media_cleanup_status` CHECK ((`status` in (_utf8mb4'WAITING',_utf8mb4'BLOCKED',_utf8mb4'CLAIMED',_utf8mb4'PURGED')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='失败接收密文的保留期、清理租约与结果审计';
