-- Flyway baseline for product version 1.0.0.
-- Generated from a fresh MySQL 8.0 database after applying V1__init_mysql.sql
-- through V1.1.0.0.8__agent_state_retention.sql. Flyway history metadata is excluded.

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_conversation` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `conversation_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '对话唯一标识（UUID）',
  `user_id` bigint DEFAULT NULL COMMENT '所属用户ID',
  `project_id` bigint DEFAULT NULL COMMENT '关联项目ID',
  `context_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '上下文类型（project/script/storyboard）',
  `agent_type` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Agent类型（script_parser/storyboard_creator）',
  `category` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '对话分类标签',
  `context_id` bigint DEFAULT NULL COMMENT '上下文对象ID',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT '新对话' COMMENT '对话标题',
  `message_count` int DEFAULT '0' COMMENT '消息总数',
  `last_message_time` datetime DEFAULT NULL COMMENT '最后消息时间',
  `status` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '对话状态：active/closed',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `next_message_order` bigint NOT NULL DEFAULT '1',
  `agent_state_last_active_at` datetime(3) DEFAULT NULL,
  `agent_state_expired_at` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `conversation_id` (`conversation_id`) USING BTREE,
  KEY `idx_conv_project_context` (`project_id`,`context_type`,`context_id`) USING BTREE,
  KEY `idx_conv_user` (`user_id`) USING BTREE,
  KEY `idx_agent_conversation_state_retention` (`agent_state_expired_at`,`agent_state_last_active_at`,`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='Agent对话索引表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_event` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `sequence_no` bigint NOT NULL,
  `schema_version` int NOT NULL DEFAULT '1',
  `raw_event_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `raw_event_type` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `source` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `reply_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `block_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `tool_call_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `parent_tool_call_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `agent_name` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `output_type` varchar(32) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `payload_json` mediumtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `event_created_at` datetime(3) DEFAULT NULL,
  `redis_published_at` datetime(3) DEFAULT NULL,
  `publish_required` tinyint NOT NULL,
  `publish_status` varchar(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `publish_claim_owner` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `publish_claim_until` datetime(3) DEFAULT NULL,
  `next_publish_attempt_at` datetime(3) DEFAULT NULL,
  `last_publish_error` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `publish_attempts` int NOT NULL DEFAULT '0',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_event_sequence` (`run_id`,`sequence_no`),
  KEY `idx_agent_event_projection` (`run_id`,`output_type`,`sequence_no`),
  KEY `idx_agent_event_publish` (`publish_status`,`next_publish_attempt_at`,`id`),
  KEY `idx_agent_event_raw` (`run_id`,`raw_event_id`),
  CONSTRAINT `chk_agent_event_publish_attempts` CHECK ((`publish_attempts` >= 0)),
  CONSTRAINT `chk_agent_event_publish_state` CHECK ((((`publish_required` = 0) and (`publish_status` = _utf8mb4'NOT_REQUIRED')) or ((`publish_required` = 1) and (`publish_status` in (_utf8mb4'PENDING',_utf8mb4'CLAIMED',_utf8mb4'PUBLISHED'))))),
  CONSTRAINT `chk_agent_event_schema_version` CHECK ((`schema_version` >= 1)),
  CONSTRAINT `chk_agent_event_sequence` CHECK ((`sequence_no` >= 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AgentScope V2 committed event journal';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_mcp_server` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint NOT NULL COMMENT '所属用户',
  `name` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '服务名称',
  `transport` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'http/sse',
  `url` varchar(2048) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'MCP 地址',
  `headers_json` text COLLATE utf8mb4_unicode_ci COMMENT '请求头 JSON',
  `query_params_json` text COLLATE utf8mb4_unicode_ci COMMENT '查询参数 JSON',
  `enabled_tools_json` text COLLATE utf8mb4_unicode_ci COMMENT '启用工具 JSON',
  `protocol_versions_json` text COLLATE utf8mb4_unicode_ci COMMENT '协议版本 JSON',
  `timeout_seconds` int NOT NULL DEFAULT '120',
  `initialization_timeout_seconds` int NOT NULL DEFAULT '30',
  `status` int NOT NULL DEFAULT '1' COMMENT '0 禁用 1 启用',
  `last_test_status` varchar(32) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `last_test_message` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `idx_agent_mcp_user_status` (`user_id`,`status`,`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户自定义 MCP 服务';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_message` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `conversation_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '所属对话ID（UUID）',
  `role` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '消息角色：user/assistant/system/tool',
  `content` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '消息文本内容',
  `references_json` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '引用资源JSON',
  `tool_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '工具调用名称（role=tool时）',
  `tool_status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '工具执行状态：running/success/error',
  `tool_call_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '工具调用ID（关联同一次调用的发起和结果）',
  `parent_tool_call_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '父级工具调用ID（子Agent事件归属到父工具调用）',
  `reasoning_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI推理过程内容（思维链）',
  `reasoning_duration_ms` bigint DEFAULT NULL COMMENT 'AI推理耗时（毫秒）',
  `message_order` bigint NOT NULL,
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `projection_key` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_agent_message_conv_order` (`conversation_id`,`message_order`),
  UNIQUE KEY `uk_agent_message_projection_key` (`projection_key`),
  KEY `idx_msg_conversation` (`conversation_id`) USING BTREE,
  KEY `idx_agent_message_conv_run_order` (`conversation_id`,`run_id`,`message_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='Agent消息表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_model_call_usage` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `model_call_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `provider` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `model_code` varchar(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `status` varchar(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `input_tokens` bigint DEFAULT NULL,
  `output_tokens` bigint DEFAULT NULL,
  `reasoning_tokens` bigint DEFAULT NULL,
  `cache_tokens` bigint DEFAULT NULL,
  `usage_json` mediumtext COLLATE utf8mb4_unicode_ci,
  `settlement_status` varchar(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `settlement_attempts` int NOT NULL DEFAULT '0',
  `next_settlement_attempt_at` datetime(3) DEFAULT NULL,
  `settlement_claim_owner` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `settlement_claim_until` datetime(3) DEFAULT NULL,
  `downstream_settlement_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `last_settlement_error` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `started_at` datetime(3) NOT NULL,
  `finished_at` datetime(3) DEFAULT NULL,
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_usage_call` (`run_id`,`model_call_id`),
  KEY `idx_agent_usage_settlement` (`settlement_status`,`next_settlement_attempt_at`,`id`),
  KEY `idx_agent_usage_run_status` (`run_id`,`status`,`id`),
  CONSTRAINT `chk_agent_usage_settlement` CHECK ((`settlement_status` in (_utf8mb4'PENDING',_utf8mb4'CLAIMED',_utf8mb4'SETTLED'))),
  CONSTRAINT `chk_agent_usage_settlement_attempts` CHECK ((`settlement_attempts` >= 0)),
  CONSTRAINT `chk_agent_usage_status` CHECK ((`status` in (_utf8mb4'STARTED',_utf8mb4'COMPLETED',_utf8mb4'FAILED',_utf8mb4'CANCELLED'))),
  CONSTRAINT `chk_agent_usage_tokens` CHECK ((((`input_tokens` is null) or (`input_tokens` >= 0)) and ((`output_tokens` is null) or (`output_tokens` >= 0)) and ((`reasoning_tokens` is null) or (`reasoning_tokens` >= 0)) and ((`cache_tokens` is null) or (`cache_tokens` >= 0))))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AgentScope V2 model-call usage ledger';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_run` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `conversation_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_id` bigint NOT NULL,
  `project_id` bigint DEFAULT NULL,
  `agent_type` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `parent_run_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `parent_tool_call_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `agent_name` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `kernel_fingerprint` varchar(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `agent_definition_snapshot_json` mediumtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `agent_state_session_id` varchar(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `status` varchar(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `owner_instance_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `owner_epoch` bigint NOT NULL DEFAULT '0',
  `lease_until` datetime(3) DEFAULT NULL,
  `next_sequence` bigint NOT NULL DEFAULT '1',
  `terminal_sequence` bigint DEFAULT NULL,
  `terminal_output_type` varchar(32) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `cancel_requested_at` datetime(3) DEFAULT NULL,
  `cancel_broadcast_at` datetime(3) DEFAULT NULL,
  `cancel_acknowledged_at` datetime(3) DEFAULT NULL,
  `cancel_next_attempt_at` datetime(3) DEFAULT NULL,
  `waiting_reply_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `waiting_tool_call_id` varchar(128) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `waiting_tool_name` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `wait_expires_at` datetime(3) DEFAULT NULL,
  `paused_through_sequence` bigint DEFAULT NULL,
  `deadline_at` datetime(3) NOT NULL,
  `started_at` datetime(3) NOT NULL,
  `heartbeat_at` datetime(3) DEFAULT NULL,
  `finished_at` datetime(3) DEFAULT NULL,
  `error_code` varchar(64) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `error_message` text COLLATE utf8mb4_unicode_ci,
  `usage_settled` tinyint NOT NULL DEFAULT '0',
  `usage_settled_at` datetime(3) DEFAULT NULL,
  `projected_through_sequence` bigint NOT NULL DEFAULT '0',
  `projection_completed_at` datetime(3) DEFAULT NULL,
  `active_conversation_id` varchar(64) CHARACTER SET ascii COLLATE ascii_bin GENERATED ALWAYS AS ((case when ((`parent_run_id` is null) and (`status` in (_utf8mb4'RUNNING',_utf8mb4'WAITING_CONFIRMATION',_utf8mb4'WAITING_EXTERNAL',_utf8mb4'CANCEL_REQUESTED'))) then `conversation_id` end)) STORED,
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_run_id` (`run_id`),
  UNIQUE KEY `uk_agent_run_active` (`active_conversation_id`),
  UNIQUE KEY `uk_agent_run_parent_tool` (`parent_run_id`,`parent_tool_call_id`),
  KEY `idx_agent_run_conversation_status` (`conversation_id`,`status`,`id`),
  KEY `idx_agent_run_parent_status` (`parent_run_id`,`status`,`id`),
  KEY `idx_agent_run_user_status` (`user_id`,`status`,`update_time`),
  KEY `idx_agent_run_lease` (`status`,`lease_until`),
  KEY `idx_agent_run_status_deadline` (`status`,`deadline_at`,`id`),
  KEY `idx_agent_run_cancel` (`status`,`cancel_next_attempt_at`),
  CONSTRAINT `chk_agent_run_parent_identity` CHECK ((((`parent_run_id` is null) and (`parent_tool_call_id` is null)) or ((`parent_run_id` is not null) and (`parent_tool_call_id` is not null) and (`agent_name` is not null)))),
  CONSTRAINT `chk_agent_run_status` CHECK ((`status` in (_utf8mb4'RUNNING',_utf8mb4'WAITING_CONFIRMATION',_utf8mb4'WAITING_EXTERNAL',_utf8mb4'CANCEL_REQUESTED',_utf8mb4'COMPLETED',_utf8mb4'FAILED',_utf8mb4'CANCELLED'))),
  CONSTRAINT `chk_agent_run_usage_settled` CHECK ((`usage_settled` in (0,1)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AgentScope V2 durable run';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_state` (
  `session_id` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `state_key` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `item_index` int NOT NULL DEFAULT '0',
  `state_data` longtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updated_at` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (`session_id`,`state_key`,`item_index`),
  KEY `idx_agent_state_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_state_cleanup_policy` (
  `id` bigint NOT NULL,
  `cleanup_interval_days` int NOT NULL DEFAULT '1',
  `retention_days` int NOT NULL DEFAULT '30',
  `next_cleanup_at` datetime(3) NOT NULL,
  `cleanup_lease_owner` varchar(160) CHARACTER SET ascii COLLATE ascii_bin DEFAULT NULL,
  `cleanup_lease_until` datetime(3) DEFAULT NULL,
  `last_cleanup_at` datetime(3) DEFAULT NULL,
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (`id`),
  CONSTRAINT `chk_agent_state_cleanup_interval` CHECK ((`cleanup_interval_days` between 1 and 365)),
  CONSTRAINT `chk_agent_state_retention_days` CHECK ((`retention_days` between 1 and 3650))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

INSERT INTO `afv_agent_state_cleanup_policy` VALUES (1,1,30,'2026-08-03 02:11:48.390',NULL,NULL,NULL,'2026-08-03 02:11:48.390','2026-08-03 02:11:48.390');
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_workspace_config` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `backend_type` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'database' COMMENT 'database/local/object_storage',
  `storage_config_id` bigint DEFAULT NULL COMMENT '对象存储配置 ID',
  `local_path` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '本地工作空间根目录',
  `migration_status` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'idle' COMMENT 'idle/copying/verifying/cutover/failed',
  `active_migration_id` bigint DEFAULT NULL COMMENT '当前迁移任务 ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `idx_agent_workspace_storage_config` (`storage_config_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间配置';
/*!40101 SET character_set_client = @saved_cs_client */;

INSERT INTO `afv_agent_workspace_config` VALUES (1,'database',NULL,NULL,'idle',NULL,'2026-08-03 02:11:46','2026-08-03 02:11:46',0x00);
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_workspace_entry` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `namespace_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '命名空间 SHA-256',
  `namespace_key` varchar(1024) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'AgentScope 命名空间',
  `item_hash` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '文件键 SHA-256',
  `item_key` varchar(2048) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '工作空间文件键',
  `backend_type` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '正文所在后端',
  `storage_config_id` bigint DEFAULT NULL COMMENT '正文所在对象存储配置 ID',
  `local_path` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '正文所在本地存储根目录',
  `content_ref` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '本地相对路径或对象键',
  `payload` longtext COLLATE utf8mb4_unicode_ci COMMENT '数据库后端正文 JSON',
  `content_sha256` char(64) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '正文 SHA-256',
  `content_size` bigint NOT NULL DEFAULT '0' COMMENT '正文字节数',
  `version` bigint NOT NULL DEFAULT '1' COMMENT 'CAS 版本',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_workspace_entry` (`namespace_hash`,`item_hash`,`deleted`),
  KEY `idx_agent_workspace_namespace` (`namespace_hash`,`id`),
  KEY `idx_agent_workspace_backend` (`backend_type`,`storage_config_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间条目';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_workspace_migration` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `source_backend_type` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `source_storage_config_id` bigint DEFAULT NULL,
  `source_local_path` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `target_backend_type` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `target_storage_config_id` bigint DEFAULT NULL,
  `target_local_path` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `status` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'copying/verifying/cutover/completed/failed/rolled_back',
  `total_count` bigint NOT NULL DEFAULT '0',
  `copied_count` bigint NOT NULL DEFAULT '0',
  `failed_count` bigint NOT NULL DEFAULT '0',
  `error_message` text COLLATE utf8mb4_unicode_ci,
  `started_at` datetime DEFAULT NULL,
  `finished_at` datetime DEFAULT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `idx_agent_workspace_migration_status` (`status`,`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间迁移任务';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_agent_workspace_migration_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `migration_id` bigint NOT NULL,
  `entry_id` bigint NOT NULL,
  `source_backend_type` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `source_storage_config_id` bigint DEFAULT NULL,
  `source_local_path` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `source_content_ref` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `source_payload` longtext COLLATE utf8mb4_unicode_ci,
  `target_backend_type` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `target_storage_config_id` bigint DEFAULT NULL,
  `target_local_path` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `target_content_ref` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `target_payload` longtext COLLATE utf8mb4_unicode_ci,
  `content_sha256` char(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `content_size` bigint NOT NULL DEFAULT '0',
  `status` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'copied',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `deleted` bit(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_agent_workspace_migration_item` (`migration_id`,`entry_id`,`deleted`),
  KEY `idx_agent_workspace_migration_item_entry` (`entry_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='智能体工作空间迁移明细';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_ai_model` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型显示名称',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型代码标识（如 deepseek-chat、qwen-vl-max）',
  `model_protocol` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '模型协议标识',
  `capability_preset_code` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '模型能力预设代码；NULL 表示自定义能力配置',
  `model_type` int NOT NULL COMMENT '模型类型：1-文本对话 2-图片生成 3-视频生成',
  `icon` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '模型图标URL',
  `description` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '模型描述说明',
  `sort` int DEFAULT '0' COMMENT '排列顺序',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `config` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '模型特定配置JSON（temperature、top_p等）',
  `default_model` tinyint DEFAULT '0' COMMENT '是否为默认模型',
  `max_concurrency` int DEFAULT '5' COMMENT '最大并发请求数',
  `api_config_id` bigint DEFAULT NULL COMMENT '关联API配置ID',
  `support_vision` tinyint DEFAULT '0' COMMENT '是否支持视觉理解（传图片）',
  `multimodal_input_types` json NOT NULL COMMENT '支持的多模态输入类型：image/video/audio/file',
  `multimodal_input_transports` json NOT NULL COMMENT '各输入类型支持的传输方式：url/base64',
  `support_reasoning` tinyint DEFAULT '0' COMMENT '是否支持深度思考（reasoning）',
  `reasoning_effort_levels` json NOT NULL COMMENT '思考等级列表，按能力从高到低排序',
  `context_window` int DEFAULT NULL COMMENT '上下文窗口大小（token数）',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `deleted_id` bigint NOT NULL DEFAULT '0' COMMENT '逻辑删除隔离标识，0-未删除，删除后为记录ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_api_config_code` (`api_config_id`,`code`,`deleted_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='AI模型表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_api_config` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '配置名称',
  `platform` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '平台标识：deepseek/dashscope/openai_compatible/ollama/anthropic/vertex_ai',
  `text_protocol` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '文本模型默认请求协议',
  `image_protocol` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '图片模型默认请求协议',
  `video_protocol` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '视频模型默认请求协议',
  `api_type` int DEFAULT NULL COMMENT 'API类型：1-文本对话 2-图片生成 3-视频生成',
  `api_url` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'API接口地址',
  `auto_append_v1_path` tinyint NOT NULL DEFAULT '1' COMMENT 'OpenAI兼容请求是否自动补充/v1路径',
  `proxy_type` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '出站代理类型：none/http/socks5',
  `proxy_host` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '出站代理主机',
  `proxy_port` int DEFAULT NULL COMMENT '出站代理端口',
  `proxy_username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '出站代理认证用户名',
  `proxy_password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '出站代理认证密码',
  `api_key` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'API密钥',
  `app_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '应用ID（部分平台需要）',
  `app_secret` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '应用密钥/服务账号 JSON Key（部分平台需要，如 Vertex AI Service Account）',
  `model_id` bigint DEFAULT NULL COMMENT '关联模型ID',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `remark` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '备注说明',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='API配置表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_asset` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_id` bigint DEFAULT NULL COMMENT '创建者用户ID',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目ID',
  `type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '资产类型：character/scene/prop/vehicle/building/costume/effect',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '资产名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '资产描述',
  `cover_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '封面图URL',
  `properties` json DEFAULT NULL COMMENT '动态属性JSON（如角色的appearance、age等）',
  `tags` json DEFAULT NULL COMMENT '标签列表JSON',
  `source_type` int DEFAULT '1' COMMENT '来源类型：1-用户上传 2-AI生成',
  `ai_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI生成时使用的提示词',
  `owner_type` int DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint DEFAULT NULL COMMENT '拥有者ID',
  `status` int DEFAULT '1' COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_asset_project` (`project_id`) USING BTREE,
  KEY `idx_asset_owner` (`owner_type`,`owner_id`) USING BTREE,
  KEY `idx_asset_type` (`project_id`,`type`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='资产表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_asset_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `asset_id` bigint NOT NULL COMMENT '所属主资产ID',
  `item_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '子资产类型：front/side/back/detail/expression/pose/variant/original',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '子资产名称',
  `image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '图片URL',
  `thumbnail_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '缩略图URL',
  `properties` json DEFAULT NULL COMMENT '动态属性JSON',
  `sort_order` int DEFAULT '0' COMMENT '排列顺序',
  `source_type` int DEFAULT '1' COMMENT '来源类型：1-用户上传 2-AI生成',
  `ai_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI生成时使用的提示词',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_asset_item_asset` (`asset_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='子资产表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_image_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` bigint NOT NULL COMMENT '所属生图任务ID',
  `platform_task_id` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '平台侧任务ID',
  `image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '生成的图片URL',
  `thumbnail_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '缩略图URL',
  `width` int DEFAULT NULL COMMENT '图片宽度（像素）',
  `height` int DEFAULT NULL COMMENT '图片高度（像素）',
  `file_size` bigint DEFAULT NULL COMMENT '文件大小（字节）',
  `status` int DEFAULT '0' COMMENT '状态：0-生成中 1-成功 2-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '失败错误信息',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_image_item_task` (`task_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='生图条目表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_image_task` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '任务唯一标识',
  `user_id` bigint NOT NULL COMMENT '发起用户ID',
  `project_id` bigint DEFAULT NULL COMMENT '关联项目ID',
  `prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '生图提示词',
  `prompt_template_id` bigint DEFAULT NULL COMMENT '提示词模板ID',
  `ref_image_urls` json DEFAULT NULL COMMENT '参考图片URL列表JSON',
  `ratio` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '画面比例（如16:9）',
  `resolution` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '分辨率（如1920x1080）',
  `aspect_ratio` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '宽高比描述',
  `width` int DEFAULT NULL COMMENT '图片宽度（像素）',
  `height` int DEFAULT NULL COMMENT '图片高度（像素）',
  `count` int DEFAULT '1' COMMENT '生成数量',
  `success_count` int DEFAULT '0' COMMENT '已成功生成数量',
  `status` int DEFAULT '0' COMMENT '任务状态：0-排队中 1-处理中 2-已完成 3-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '失败错误信息',
  `model_id` bigint DEFAULT NULL COMMENT '使用的AI模型ID',
  `category` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '任务分类标签',
  `owner_type` int DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint DEFAULT NULL COMMENT '拥有者ID',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `task_id` (`task_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='生图任务表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_project` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '项目名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '项目描述',
  `cover_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '项目封面图URL',
  `scope` int DEFAULT '2' COMMENT '可见范围：1-公开 2-私有 3-仅团队可见',
  `owner_type` int NOT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NOT NULL COMMENT '拥有者ID',
  `status` int DEFAULT '0' COMMENT '状态：0-筹备中 1-进行中 2-已完成 3-已归档',
  `properties` json DEFAULT NULL COMMENT '扩展配置JSON',
  `art_style` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '画风key（预设key或custom）',
  `art_style_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '画风中文描述',
  `art_style_image_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '画风英文提示词',
  `art_style_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '画风参考图路径',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='视频项目表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_project_member` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `project_id` bigint NOT NULL COMMENT '所属项目ID',
  `user_id` bigint NOT NULL COMMENT '成员用户ID',
  `role` int NOT NULL DEFAULT '3' COMMENT '成员角色：1-拥有者 2-管理员 3-普通成员',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_project_user` (`project_id`,`user_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='项目成员表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_script` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `project_id` bigint NOT NULL COMMENT '所属项目ID',
  `title` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '剧本标题',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '剧本正文内容（格式化后）',
  `raw_content` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '剧本原始内容（用户粘贴的原文）',
  `total_episodes` int DEFAULT '0' COMMENT '总集数',
  `story_synopsis` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '故事梗概',
  `characters_json` json DEFAULT NULL COMMENT '角色列表JSON',
  `source_type` int DEFAULT '0' COMMENT '来源类型：0-手动创建 1-文件导入 2-AI生成',
  `parsing_status` int DEFAULT '0' COMMENT '解析状态：0-未解析 1-解析中 2-解析完成 3-解析失败',
  `parsing_progress` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '解析进度描述',
  `summary` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI生成的剧本摘要',
  `genre` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '剧本类型/题材',
  `target_audience` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '目标受众',
  `duration_estimate` int DEFAULT NULL COMMENT '预估总时长（分钟）',
  `scope` int DEFAULT '3' COMMENT '可见范围：1-公开 2-私有 3-仅团队可见',
  `owner_type` int DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint DEFAULT NULL COMMENT '拥有者ID',
  `ai_generated` tinyint DEFAULT '0' COMMENT '是否由AI生成',
  `version` int DEFAULT '0' COMMENT '乐观锁版本号',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_script_project` (`project_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='剧本表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_script_episode` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `script_id` bigint NOT NULL COMMENT '所属剧本ID',
  `episode_number` int DEFAULT NULL COMMENT '集号（从1开始）',
  `title` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '本集标题',
  `synopsis` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '本集剧情梗概',
  `raw_content` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '本集原始剧本内容',
  `duration_estimate` int DEFAULT NULL COMMENT '预估时长（分钟）',
  `total_scenes` int DEFAULT '0' COMMENT '本集总场次数',
  `source_type` int DEFAULT '0' COMMENT '来源类型：0-AI解析 1-手动添加',
  `sort_order` int DEFAULT '0' COMMENT '排列顺序',
  `parsing_status` int DEFAULT '0' COMMENT '解析状态：0-未解析 1-解析中 2-解析完成 3-解析失败',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `version` int DEFAULT '0' COMMENT '乐观锁版本号',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_episode_script` (`script_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='分集剧本表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_script_scene_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `episode_id` bigint NOT NULL COMMENT '所属分集ID',
  `script_id` bigint NOT NULL COMMENT '所属剧本ID',
  `scene_number` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '场次编号（如1-1表示第1集第1场）',
  `scene_heading` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '场景标头（如"内景 客厅 夜"）',
  `location` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '场景地点',
  `time_of_day` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '时间段：日/夜/黄昏/清晨等',
  `int_ext` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '内外景标识：内景/外景/内外景',
  `characters` json DEFAULT NULL COMMENT '出场角色名列表JSON',
  `character_asset_ids` json DEFAULT NULL COMMENT '出场角色资产ID列表JSON',
  `scene_asset_id` bigint DEFAULT NULL COMMENT '场景资产ID',
  `prop_asset_ids` json DEFAULT NULL COMMENT '道具资产ID列表JSON',
  `scene_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '场景氛围/环境描述',
  `dialogues` json DEFAULT NULL COMMENT '对白/动作元素列表JSON',
  `sort_order` int DEFAULT '0' COMMENT '排列顺序',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `version` int DEFAULT '0' COMMENT '乐观锁版本号',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_script_scene_episode` (`episode_id`) USING BTREE,
  KEY `idx_script_scene_script` (`script_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='剧本分场次表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_storage_config` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '配置名称',
  `type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '存储类型：local / aliyun_oss / tencent_cos / s3',
  `provider` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'S3兼容厂商：generic_s3/aliyun_oss/tencent_cos/qiniu_kodo/ctyun_zos/minio',
  `endpoint` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'OSS 端点地址',
  `bucket_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'OSS 存储桶名称',
  `access_key` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'OSS Access Key',
  `secret_key` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'OSS Secret Key',
  `region` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '区域',
  `base_path` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '存储根路径（本地为磁盘路径，OSS 为 key 前缀）',
  `custom_domain` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '自定义域名（CDN 域名等）',
  `options` json DEFAULT NULL COMMENT '厂商扩展配置JSON',
  `is_default` tinyint NOT NULL DEFAULT '0' COMMENT '是否为默认存储配置',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `remark` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '备注',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='存储配置表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_storyboard` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `project_id` bigint DEFAULT NULL COMMENT '所属项目ID',
  `script_id` bigint DEFAULT NULL COMMENT '关联剧本ID',
  `title` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '分镜标题',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '分镜描述',
  `custom_columns` json DEFAULT NULL COMMENT '自定义列配置JSON',
  `scope` int DEFAULT '3' COMMENT '可见范围：1-公开 2-私有 3-仅团队可见',
  `owner_type` int DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint DEFAULT NULL COMMENT '拥有者ID',
  `total_duration` int DEFAULT NULL COMMENT '预估总时长（秒）',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_storyboard_project` (`project_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='分镜脚本表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_storyboard_episode` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `storyboard_id` bigint NOT NULL COMMENT '所属分镜ID',
  `script_episode_id` bigint DEFAULT NULL COMMENT '关联的剧本分集ID',
  `episode_number` int DEFAULT NULL COMMENT '集号',
  `title` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '集标题',
  `synopsis` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '本集梗概',
  `sort_order` int DEFAULT '0' COMMENT '排列顺序',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint DEFAULT '0' COMMENT '逻辑删除标志',
  `deleted_id` bigint NOT NULL DEFAULT '0' COMMENT '逻辑删除隔离标识，0-未删除，删除后为记录ID',
  `create_time` datetime DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  `composed_video_url` varchar(512) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '本集合成视频URL',
  `compose_status` tinyint NOT NULL DEFAULT '0' COMMENT '合成状态: 0未开始 1合成中 2已完成 3失败',
  `compose_error_msg` varchar(1024) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '合成失败原因',
  `composed_at` datetime DEFAULT NULL COMMENT '合成完成时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_sb_episode_script_episode` (`storyboard_id`,`script_episode_id`,`deleted_id`) USING BTREE,
  KEY `idx_sb_episode_storyboard` (`storyboard_id`) USING BTREE,
  KEY `idx_sb_episode_script_episode` (`script_episode_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='分镜集表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_storyboard_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `storyboard_id` bigint NOT NULL COMMENT '所属分镜ID',
  `storyboard_episode_id` bigint DEFAULT NULL COMMENT '所属分镜集ID',
  `storyboard_scene_id` bigint DEFAULT NULL COMMENT '所属分镜场次ID',
  `sort_order` int DEFAULT '0' COMMENT '排列顺序',
  `shot_number` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '镜号',
  `auto_shot_number` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '自动编号（系统生成）',
  `image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '用户上传参考图片URL',
  `reference_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '外部参考图片URL',
  `video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '视频URL（最终成品）',
  `generated_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'AI生成的图片URL',
  `first_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '首帧参考图片URL',
  `last_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '尾帧参考图片URL',
  `first_frame_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI生成首帧时使用的提示词',
  `last_frame_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI生成尾帧时使用的提示词',
  `generated_video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'AI生成的视频URL',
  `shot_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '景别：远景/全景/中景/近景/特写',
  `duration` decimal(10,2) DEFAULT NULL COMMENT '预估时长（秒）',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '画面内容描述',
  `scene_expectation` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '画面期望描述（AI生图提示）',
  `sound` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '声音描述',
  `dialogue` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '台词/旁白',
  `sound_effect` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '音效',
  `music` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '配乐建议',
  `camera_movement` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '镜头运动：推/拉/摇/移/跟/升/降',
  `camera_angle` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '镜头角度：平视/俯视/仰视',
  `camera_equipment` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '摄像机装备',
  `focal_length` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '镜头焦段',
  `transition` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '转场效果：切/淡入/淡出/溶/划',
  `character_ids` json DEFAULT NULL COMMENT '出场角色子资产ID列表 JSON (List<Long> of AssetItem.id)',
  `scene_asset_item_id` bigint DEFAULT NULL COMMENT '场景子资产ID (AssetItem.id)',
  `prop_ids` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '道具子资产ID列表 JSON (List<Long> of AssetItem.id)',
  `remark` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '备注',
  `custom_data` json DEFAULT NULL COMMENT '自定义扩展数据JSON',
  `ai_generated` tinyint DEFAULT '0' COMMENT '是否由AI生成',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `video_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'AI生成视频时使用的提示词（保存以便复用和手动调整）',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_sb_item_storyboard` (`storyboard_id`) USING BTREE,
  KEY `idx_sb_item_scene` (`storyboard_scene_id`) USING BTREE,
  KEY `idx_sb_item_episode` (`storyboard_episode_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='分镜条目表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_storyboard_scene` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `episode_id` bigint NOT NULL COMMENT '所属分镜集ID',
  `storyboard_id` bigint NOT NULL COMMENT '所属分镜ID（冗余）',
  `scene_number` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '场次编号',
  `scene_heading` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '场景标头',
  `location` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '场景地点',
  `time_of_day` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '时间段',
  `int_ext` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '内外景标识',
  `sort_order` int DEFAULT '0' COMMENT '排列顺序',
  `status` int DEFAULT '0' COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_sb_scene_episode` (`episode_id`) USING BTREE,
  KEY `idx_sb_scene_storyboard` (`storyboard_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='分镜场次表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_system_config` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `config_key` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '配置键',
  `config_value` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '配置值',
  `remark` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '备注',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_config_key` (`config_key`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='系统配置表';
/*!40101 SET character_set_client = @saved_cs_client */;

INSERT INTO `afv_system_config` VALUES (1,'allow_register','false','是否允许公开注册',0,'2026-08-03 02:11:43','2026-08-03 02:11:43');
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_team` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '团队名称',
  `logo` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '团队LOGO图片URL',
  `description` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '团队描述',
  `owner_user_id` bigint NOT NULL COMMENT '创建者用户ID',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='团队表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_team_member` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `team_id` bigint NOT NULL COMMENT '所属团队ID',
  `user_id` bigint NOT NULL COMMENT '成员用户ID',
  `role` int NOT NULL DEFAULT '3' COMMENT '角色：1-创建者 2-管理员 3-普通成员',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `join_time` datetime DEFAULT NULL COMMENT '加入时间',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_team_user` (`team_id`,`user_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='团队成员表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_video_item` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` bigint NOT NULL COMMENT '所属生视频任务ID',
  `platform_task_id` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '平台侧任务ID',
  `video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '生成的视频URL',
  `cover_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '视频封面图URL',
  `duration` int DEFAULT NULL COMMENT '视频时长（秒）',
  `file_size` bigint DEFAULT NULL COMMENT '文件大小（字节）',
  `status` int DEFAULT '0' COMMENT '状态：0-生成中 1-成功 2-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '失败错误信息',
  `first_frame_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '视频首帧图片URL',
  `last_frame_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '视频尾帧图片URL',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_video_item_task` (`task_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='生视频条目表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `afv_video_task` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '任务唯一标识',
  `user_id` bigint NOT NULL COMMENT '发起用户ID',
  `project_id` bigint DEFAULT NULL COMMENT '关联项目ID',
  `prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '生视频提示词',
  `prompt_template_id` bigint DEFAULT NULL COMMENT '提示词模板ID',
  `generate_mode` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '生成模式：text2video/image2video',
  `first_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '首帧参考图片URL',
  `last_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '尾帧参考图片URL',
  `reference_image_urls` json DEFAULT NULL COMMENT '参考图片URL列表JSON',
  `reference_video_urls` json DEFAULT NULL COMMENT '参考视频URL列表 JSON',
  `reference_audio_urls` json DEFAULT NULL COMMENT '参考音频URL列表 JSON',
  `ratio` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '画面比例（如16:9）',
  `resolution` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '分辨率（如1920x1080）',
  `duration` int DEFAULT NULL COMMENT '视频时长（秒）',
  `watermark` tinyint DEFAULT '0' COMMENT '是否添加水印',
  `generate_audio` tinyint DEFAULT '0' COMMENT '是否生成配音',
  `seed` bigint DEFAULT NULL COMMENT '随机种子（用于复现）',
  `camera_fixed` tinyint DEFAULT '0' COMMENT '是否固定镜头',
  `count` int DEFAULT '1' COMMENT '生成数量',
  `success_count` int DEFAULT '0' COMMENT '已成功生成数量',
  `status` int DEFAULT '0' COMMENT '任务状态：0-排队中 1-处理中 2-已完成 3-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '失败错误信息',
  `model_id` bigint DEFAULT NULL COMMENT '使用的AI模型ID',
  `category` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '任务分类标签',
  `owner_type` int DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint DEFAULT NULL COMMENT '拥有者ID',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `task_id` (`task_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='生视频任务表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_role` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '角色名称',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '角色代码标识（如 admin、user）',
  `sort` int DEFAULT '0' COMMENT '排列顺序',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `remark` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '备注说明',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `code` (`code`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='角色表';
/*!40101 SET character_set_client = @saved_cs_client */;

INSERT INTO `sys_role` VALUES (1,'超级管理员','admin',1,1,'系统超级管理员',0,'2026-04-16 16:03:20','2026-04-16 16:03:20'),(2,'普通用户','user',2,1,'默认用户角色',0,'2026-04-16 16:03:20','2026-04-16 16:03:20');
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_user` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `username` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '登录用户名（唯一）',
  `password` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '登录密码（BCrypt加密）',
  `nickname` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '用户昵称',
  `avatar` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '头像URL',
  `email` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '邮箱地址',
  `phone` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '手机号码',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用 1-启用',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `username` (`username`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='用户表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_user_role` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `role_id` bigint NOT NULL COMMENT '角色ID',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_user_role` (`user_id`,`role_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='用户角色关联表';
/*!40101 SET character_set_client = @saved_cs_client */;

/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
