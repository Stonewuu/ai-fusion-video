CREATE TABLE `afv_agent_workspace_config` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `backend_type` varchar(32) NOT NULL DEFAULT 'database' COMMENT 'database/local/object_storage',
  `storage_config_id` bigint DEFAULT NULL COMMENT '对象存储配置 ID',
  `local_path` varchar(512) DEFAULT NULL COMMENT '本地工作空间根目录',
  `migration_status` varchar(32) NOT NULL DEFAULT 'idle' COMMENT 'idle/copying/verifying/cutover/failed',
  `active_migration_id` bigint DEFAULT NULL COMMENT '当前迁移任务 ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `idx_agent_workspace_storage_config` (`storage_config_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间配置';

INSERT INTO `afv_agent_workspace_config`
  (`id`, `backend_type`, `migration_status`, `create_time`, `update_time`, `deleted`)
VALUES (1, 'database', 'idle', NOW(), NOW(), b'0');

CREATE TABLE `afv_agent_workspace_entry` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `namespace_hash` char(64) NOT NULL COMMENT '命名空间 SHA-256',
  `namespace_key` varchar(1024) NOT NULL COMMENT 'AgentScope 命名空间',
  `item_hash` char(64) NOT NULL COMMENT '文件键 SHA-256',
  `item_key` varchar(2048) NOT NULL COMMENT '工作空间文件键',
  `backend_type` varchar(32) NOT NULL COMMENT '正文所在后端',
  `storage_config_id` bigint DEFAULT NULL COMMENT '正文所在对象存储配置 ID',
  `local_path` varchar(512) DEFAULT NULL COMMENT '正文所在本地存储根目录',
  `content_ref` varchar(1024) DEFAULT NULL COMMENT '本地相对路径或对象键',
  `payload` longtext COMMENT '数据库后端正文 JSON',
  `content_sha256` char(64) NOT NULL COMMENT '正文 SHA-256',
  `content_size` bigint NOT NULL DEFAULT 0 COMMENT '正文字节数',
  `version` bigint NOT NULL DEFAULT 1 COMMENT 'CAS 版本',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_workspace_entry` (`namespace_hash`, `item_hash`, `deleted`),
  KEY `idx_agent_workspace_namespace` (`namespace_hash`, `id`),
  KEY `idx_agent_workspace_backend` (`backend_type`, `storage_config_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间条目';

CREATE TABLE `afv_agent_workspace_migration` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `source_backend_type` varchar(32) NOT NULL,
  `source_storage_config_id` bigint DEFAULT NULL,
  `source_local_path` varchar(512) DEFAULT NULL,
  `target_backend_type` varchar(32) NOT NULL,
  `target_storage_config_id` bigint DEFAULT NULL,
  `target_local_path` varchar(512) DEFAULT NULL,
  `status` varchar(32) NOT NULL COMMENT 'copying/verifying/cutover/completed/failed/rolled_back',
  `total_count` bigint NOT NULL DEFAULT 0,
  `copied_count` bigint NOT NULL DEFAULT 0,
  `failed_count` bigint NOT NULL DEFAULT 0,
  `error_message` text,
  `started_at` datetime DEFAULT NULL,
  `finished_at` datetime DEFAULT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `idx_agent_workspace_migration_status` (`status`, `id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间迁移任务';

CREATE TABLE `afv_agent_workspace_migration_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `migration_id` bigint NOT NULL,
  `entry_id` bigint NOT NULL,
  `source_backend_type` varchar(32) NOT NULL,
  `source_storage_config_id` bigint DEFAULT NULL,
  `source_local_path` varchar(512) DEFAULT NULL,
  `source_content_ref` varchar(1024) DEFAULT NULL,
  `source_payload` longtext,
  `target_backend_type` varchar(32) NOT NULL,
  `target_storage_config_id` bigint DEFAULT NULL,
  `target_local_path` varchar(512) DEFAULT NULL,
  `target_content_ref` varchar(1024) DEFAULT NULL,
  `target_payload` longtext,
  `content_sha256` char(64) NOT NULL,
  `content_size` bigint NOT NULL DEFAULT 0,
  `status` varchar(32) NOT NULL DEFAULT 'copied',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_workspace_migration_item` (`migration_id`, `entry_id`, `deleted`),
  KEY `idx_agent_workspace_migration_item_entry` (`entry_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间迁移明细';

CREATE TABLE `afv_agent_mcp_server` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint NOT NULL COMMENT '所属用户',
  `name` varchar(128) NOT NULL COMMENT '服务名称',
  `transport` varchar(32) NOT NULL COMMENT 'http/sse',
  `url` varchar(2048) NOT NULL COMMENT 'MCP 地址',
  `headers_json` text COMMENT '请求头 JSON',
  `query_params_json` text COMMENT '查询参数 JSON',
  `enabled_tools_json` text COMMENT '启用工具 JSON',
  `protocol_versions_json` text COMMENT '协议版本 JSON',
  `timeout_seconds` int NOT NULL DEFAULT 120,
  `initialization_timeout_seconds` int NOT NULL DEFAULT 30,
  `status` int NOT NULL DEFAULT 1 COMMENT '0 禁用 1 启用',
  `last_test_status` varchar(32) DEFAULT NULL,
  `last_test_message` varchar(1024) DEFAULT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `idx_agent_mcp_user_status` (`user_id`, `status`, `id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户自定义 MCP 服务';
