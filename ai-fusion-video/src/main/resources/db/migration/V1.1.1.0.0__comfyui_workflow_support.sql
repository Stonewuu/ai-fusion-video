CREATE TABLE `afv_comfyui_workflow` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `api_config_id` bigint NOT NULL COMMENT 'ComfyUI 接口配置标识',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '工作流显示名称',
  `code` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '工作流稳定标识',
  `model_type` int NOT NULL COMMENT '模型类型：2-图片，3-视频',
  `description` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '工作流说明',
  `active_version_id` bigint DEFAULT NULL COMMENT '当前发布的工作流版本标识',
  `status` int NOT NULL DEFAULT '1' COMMENT '状态：0-禁用，1-启用',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标记',
  `deleted_id` bigint NOT NULL DEFAULT '0' COMMENT '逻辑删除隔离标识',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_comfyui_workflow_code` (`api_config_id`, `code`, `deleted_id`) USING BTREE,
  KEY `idx_comfyui_workflow_api_config` (`api_config_id`, `status`) USING BTREE,
  KEY `idx_comfyui_workflow_active_version` (`active_version_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='ComfyUI 工作流';

CREATE TABLE `afv_comfyui_workflow_version` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `workflow_id` bigint NOT NULL COMMENT '工作流标识',
  `version_no` int NOT NULL COMMENT '单调递增的版本号',
  `ui_workflow_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '可选的 ComfyUI 界面格式工作流',
  `api_workflow_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'ComfyUI 接口格式工作流',
  `input_bindings_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '平台输入绑定',
  `output_bindings_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '显式输出绑定',
  `required_nodes_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '必需节点类列表',
  `workflow_hash` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '规范化执行定义的哈希值',
  `validation_status` int NOT NULL DEFAULT '0' COMMENT '校验状态：0-未校验，1-有效，2-无效',
  `validation_message` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '校验详情',
  `test_status` int NOT NULL DEFAULT '0' COMMENT '试运行状态：0-未试运行，1-通过，2-失败',
  `test_message` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '试运行详情',
  `last_test_time` datetime DEFAULT NULL COMMENT '最后试运行时间',
  `published` tinyint NOT NULL DEFAULT '0' COMMENT '此版本是否曾经发布',
  `deleted` tinyint NOT NULL DEFAULT '0' COMMENT '逻辑删除标记',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_comfyui_workflow_version` (`workflow_id`, `version_no`) USING BTREE,
  KEY `idx_comfyui_workflow_version_status` (`workflow_id`, `validation_status`, `test_status`, `published`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='不可变的 ComfyUI 工作流版本';

ALTER TABLE `afv_ai_model`
  ADD COLUMN `comfyui_workflow_id` bigint DEFAULT NULL COMMENT '关联的 ComfyUI 工作流标识' AFTER `api_config_id`,
  ADD KEY `idx_ai_model_comfyui_workflow` (`comfyui_workflow_id`);

ALTER TABLE `afv_image_task`
  ADD COLUMN `workflow_version_id` bigint DEFAULT NULL COMMENT '固定的 ComfyUI 工作流版本标识' AFTER `model_id`,
  ADD KEY `idx_image_task_workflow_version` (`workflow_version_id`);

ALTER TABLE `afv_video_task`
  ADD COLUMN `workflow_version_id` bigint DEFAULT NULL COMMENT '固定的 ComfyUI 工作流版本标识' AFTER `model_id`,
  ADD KEY `idx_video_task_workflow_version` (`workflow_version_id`);
