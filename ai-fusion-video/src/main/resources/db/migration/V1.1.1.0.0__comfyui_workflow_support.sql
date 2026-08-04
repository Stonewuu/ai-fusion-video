CREATE TABLE `afv_comfyui_workflow` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
  `api_config_id` bigint NOT NULL COMMENT 'ComfyUI API configuration ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Workflow display name',
  `code` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Stable workflow code',
  `model_type` int NOT NULL COMMENT 'Model type: 2-image, 3-video',
  `description` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Workflow description',
  `active_version_id` bigint DEFAULT NULL COMMENT 'Currently published workflow version ID',
  `status` int NOT NULL DEFAULT '1' COMMENT 'Status: 0-disabled, 1-enabled',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT 'Logical delete flag',
  `deleted_id` bigint NOT NULL DEFAULT '0' COMMENT 'Logical delete isolation ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_comfyui_workflow_code` (`api_config_id`, `code`, `deleted_id`) USING BTREE,
  KEY `idx_comfyui_workflow_api_config` (`api_config_id`, `status`) USING BTREE,
  KEY `idx_comfyui_workflow_active_version` (`active_version_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='ComfyUI workflow';

CREATE TABLE `afv_comfyui_workflow_version` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
  `workflow_id` bigint NOT NULL COMMENT 'Workflow ID',
  `version_no` int NOT NULL COMMENT 'Monotonically increasing version number',
  `ui_workflow_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Optional ComfyUI UI-format source',
  `api_workflow_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'ComfyUI API-format workflow',
  `input_bindings_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Platform input bindings',
  `output_bindings_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Explicit output bindings',
  `required_nodes_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Required node class list',
  `workflow_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT 'SHA-256 of canonical execution definition',
  `validation_status` int NOT NULL DEFAULT '0' COMMENT 'Validation: 0-unvalidated, 1-valid, 2-invalid',
  `validation_message` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Validation details',
  `test_status` int NOT NULL DEFAULT '0' COMMENT 'Execution test: 0-untested, 1-passed, 2-failed',
  `test_message` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Execution test details',
  `last_test_time` datetime DEFAULT NULL COMMENT 'Last execution test time',
  `published` tinyint NOT NULL DEFAULT '0' COMMENT 'Whether this version has ever been published',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT 'Logical delete flag',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_comfyui_workflow_version` (`workflow_id`, `version_no`) USING BTREE,
  KEY `idx_comfyui_workflow_version_status` (`workflow_id`, `validation_status`, `test_status`, `published`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='Immutable ComfyUI workflow version';

ALTER TABLE `afv_ai_model`
  ADD COLUMN `comfyui_workflow_id` bigint DEFAULT NULL COMMENT 'Associated ComfyUI workflow ID' AFTER `api_config_id`,
  ADD KEY `idx_ai_model_comfyui_workflow` (`comfyui_workflow_id`);

ALTER TABLE `afv_image_task`
  ADD COLUMN `workflow_version_id` bigint DEFAULT NULL COMMENT 'Pinned ComfyUI workflow version ID' AFTER `model_id`,
  ADD KEY `idx_image_task_workflow_version` (`workflow_version_id`);

ALTER TABLE `afv_video_task`
  ADD COLUMN `workflow_version_id` bigint DEFAULT NULL COMMENT 'Pinned ComfyUI workflow version ID' AFTER `model_id`,
  ADD KEY `idx_video_task_workflow_version` (`workflow_version_id`);
