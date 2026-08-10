SET NAMES utf8mb4;
-- ----------------------------
-- Table structure for afv_agent_conversation
-- ----------------------------
DROP TABLE IF EXISTS `afv_agent_conversation`;
CREATE TABLE `afv_agent_conversation`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `conversation_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '对话唯一标识（UUID）',
  `user_id` bigint NULL DEFAULT NULL COMMENT '所属用户ID',
  `project_id` bigint NULL DEFAULT NULL COMMENT '关联项目ID',
  `context_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '上下文类型（project/script/storyboard）',
  `agent_type` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'Agent类型（script_parser/storyboard_creator）',
  `category` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '对话分类标签',
  `context_id` bigint NULL DEFAULT NULL COMMENT '上下文对象ID',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '新对话' COMMENT '对话标题',
  `message_count` int NULL DEFAULT 0 COMMENT '消息总数',
  `last_message_time` datetime NULL DEFAULT NULL COMMENT '最后消息时间',
  `status` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '对话状态：active/closed',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `conversation_id`(`conversation_id` ASC) USING BTREE,
  INDEX `idx_conv_project_context`(`project_id` ASC, `context_type` ASC, `context_id` ASC) USING BTREE,
  INDEX `idx_conv_user`(`user_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = 'Agent对话索引表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_agent_conversation
-- ----------------------------

-- ----------------------------
-- Table structure for afv_agent_message
-- ----------------------------
DROP TABLE IF EXISTS `afv_agent_message`;
CREATE TABLE `afv_agent_message`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `conversation_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '所属对话ID（UUID）',
  `role` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '消息角色：user/assistant/system/tool',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '消息文本内容',
  `references_json` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '引用资源JSON',
  `tool_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '工具调用名称（role=tool时）',
  `tool_status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '工具执行状态：running/success/error',
  `tool_call_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '工具调用ID（关联同一次调用的发起和结果）',
  `parent_tool_call_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '父级工具调用ID（子Agent事件归属到父工具调用）',
  `reasoning_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI推理过程内容（思维链）',
  `reasoning_duration_ms` bigint NULL DEFAULT NULL COMMENT 'AI推理耗时（毫秒）',
  `message_order` int NOT NULL COMMENT '消息排列顺序',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_msg_conversation`(`conversation_id` ASC) USING BTREE,
  INDEX `idx_msg_conv_order`(`conversation_id` ASC, `message_order` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = 'Agent消息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_agent_message
-- ----------------------------

-- ----------------------------
-- Table structure for afv_ai_model
-- ----------------------------
DROP TABLE IF EXISTS `afv_ai_model`;
CREATE TABLE `afv_ai_model`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型显示名称',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型代码标识（如 deepseek-chat、qwen-vl-max）',
  `model_type` int NOT NULL COMMENT '模型类型：1-文本对话 2-图片生成 3-视频生成',
  `icon` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '模型图标URL',
  `description` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '模型描述说明',
  `sort` int NULL DEFAULT 0 COMMENT '排列顺序',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `config` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '模型特定配置JSON（temperature、top_p等）',
  `default_model` tinyint NULL DEFAULT 0 COMMENT '是否为默认模型',
  `max_concurrency` int NULL DEFAULT 5 COMMENT '最大并发请求数',
  `api_config_id` bigint NULL DEFAULT NULL COMMENT '关联API配置ID',
  `support_vision` tinyint NULL DEFAULT 0 COMMENT '是否支持视觉理解（传图片）',
  `support_reasoning` tinyint NULL DEFAULT 0 COMMENT '是否支持深度思考（reasoning）',
  `context_window` int NULL DEFAULT NULL COMMENT '上下文窗口大小（token数）',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `code`(`code` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = 'AI模型表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_ai_model
-- ----------------------------

-- ----------------------------
-- Table structure for afv_api_config
-- ----------------------------
DROP TABLE IF EXISTS `afv_api_config`;
CREATE TABLE `afv_api_config`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '配置名称',
  `platform` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '平台标识：deepseek/dashscope/openai_compatible/ollama/anthropic/vertex_ai',
  `api_type` int NULL DEFAULT NULL COMMENT 'API类型：1-文本对话 2-图片生成 3-视频生成',
  `api_url` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'API接口地址',
  `api_key` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'API密钥',
  `app_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '应用ID（部分平台需要）',
  `app_secret` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '应用密钥（部分平台需要）',
  `model_id` bigint NULL DEFAULT NULL COMMENT '关联模型ID',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `remark` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '备注说明',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = 'API配置表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_api_config
-- ----------------------------

-- ----------------------------
-- Table structure for afv_asset
-- ----------------------------
DROP TABLE IF EXISTS `afv_asset`;
CREATE TABLE `afv_asset`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_id` bigint NULL DEFAULT NULL COMMENT '创建者用户ID',
  `project_id` bigint NULL DEFAULT NULL COMMENT '所属项目ID',
  `type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '资产类型：character/scene/prop/vehicle/building/costume/effect',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '资产名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '资产描述',
  `cover_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '封面图URL',
  `properties` json NULL COMMENT '动态属性JSON（如角色的appearance、age等）',
  `tags` json NULL COMMENT '标签列表JSON',
  `source_type` int NULL DEFAULT 1 COMMENT '来源类型：1-用户上传 2-AI生成',
  `ai_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI生成时使用的提示词',
  `owner_type` int NULL DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NULL DEFAULT NULL COMMENT '拥有者ID',
  `status` int NULL DEFAULT 1 COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_asset_project`(`project_id` ASC) USING BTREE,
  INDEX `idx_asset_owner`(`owner_type` ASC, `owner_id` ASC) USING BTREE,
  INDEX `idx_asset_type`(`project_id` ASC, `type` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '资产表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_asset
-- ----------------------------

-- ----------------------------
-- Table structure for afv_asset_item
-- ----------------------------
DROP TABLE IF EXISTS `afv_asset_item`;
CREATE TABLE `afv_asset_item`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `asset_id` bigint NOT NULL COMMENT '所属主资产ID',
  `item_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '子资产类型：front/side/back/detail/expression/pose/variant/original',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '子资产名称',
  `image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '图片URL',
  `thumbnail_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '缩略图URL',
  `properties` json NULL COMMENT '动态属性JSON',
  `sort_order` int NULL DEFAULT 0 COMMENT '排列顺序',
  `source_type` int NULL DEFAULT 1 COMMENT '来源类型：1-用户上传 2-AI生成',
  `ai_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI生成时使用的提示词',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_asset_item_asset`(`asset_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '子资产表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_asset_item
-- ----------------------------

-- ----------------------------
-- Table structure for afv_image_item
-- ----------------------------
DROP TABLE IF EXISTS `afv_image_item`;
CREATE TABLE `afv_image_item`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` bigint NOT NULL COMMENT '所属生图任务ID',
  `platform_task_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '平台侧任务ID',
  `image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '生成的图片URL',
  `thumbnail_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '缩略图URL',
  `width` int NULL DEFAULT NULL COMMENT '图片宽度（像素）',
  `height` int NULL DEFAULT NULL COMMENT '图片高度（像素）',
  `file_size` bigint NULL DEFAULT NULL COMMENT '文件大小（字节）',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-生成中 1-成功 2-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '失败错误信息',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_image_item_task`(`task_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '生图条目表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_image_item
-- ----------------------------

-- ----------------------------
-- Table structure for afv_image_task
-- ----------------------------
DROP TABLE IF EXISTS `afv_image_task`;
CREATE TABLE `afv_image_task`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '任务唯一标识',
  `user_id` bigint NOT NULL COMMENT '发起用户ID',
  `project_id` bigint NULL DEFAULT NULL COMMENT '关联项目ID',
  `prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '生图提示词',
  `prompt_template_id` bigint NULL DEFAULT NULL COMMENT '提示词模板ID',
  `ref_image_urls` json NULL COMMENT '参考图片URL列表JSON',
  `ratio` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '画面比例（如16:9）',
  `resolution` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '分辨率（如1920x1080）',
  `aspect_ratio` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '宽高比描述',
  `width` int NULL DEFAULT NULL COMMENT '图片宽度（像素）',
  `height` int NULL DEFAULT NULL COMMENT '图片高度（像素）',
  `count` int NULL DEFAULT 1 COMMENT '生成数量',
  `success_count` int NULL DEFAULT 0 COMMENT '已成功生成数量',
  `status` int NULL DEFAULT 0 COMMENT '任务状态：0-排队中 1-处理中 2-已完成 3-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '失败错误信息',
  `model_id` bigint NULL DEFAULT NULL COMMENT '使用的AI模型ID',
  `category` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '任务分类标签',
  `owner_type` int NULL DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NULL DEFAULT NULL COMMENT '拥有者ID',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `task_id`(`task_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '生图任务表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_image_task
-- ----------------------------

-- ----------------------------
-- Table structure for afv_project
-- ----------------------------
DROP TABLE IF EXISTS `afv_project`;
CREATE TABLE `afv_project`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '项目名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '项目描述',
  `cover_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '项目封面图URL',
  `scope` int NULL DEFAULT 2 COMMENT '可见范围：1-公开 2-私有 3-仅团队可见',
  `owner_type` int NOT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NOT NULL COMMENT '拥有者ID',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-筹备中 1-进行中 2-已完成 3-已归档',
  `properties` json NULL COMMENT '扩展配置JSON',
  `art_style` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '画风key（预设key或custom）',
  `art_style_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '画风中文描述',
  `art_style_image_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '画风英文提示词',
  `art_style_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '画风参考图路径',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '视频项目表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_project
-- ----------------------------

-- ----------------------------
-- Table structure for afv_project_member
-- ----------------------------
DROP TABLE IF EXISTS `afv_project_member`;
CREATE TABLE `afv_project_member`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `project_id` bigint NOT NULL COMMENT '所属项目ID',
  `user_id` bigint NOT NULL COMMENT '成员用户ID',
  `role` int NOT NULL DEFAULT 3 COMMENT '成员角色：1-拥有者 2-管理员 3-普通成员',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_project_user`(`project_id` ASC, `user_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '项目成员表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_project_member
-- ----------------------------

-- ----------------------------
-- Table structure for afv_script
-- ----------------------------
DROP TABLE IF EXISTS `afv_script`;
CREATE TABLE `afv_script`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `project_id` bigint NOT NULL COMMENT '所属项目ID',
  `title` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '剧本标题',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '剧本正文内容（格式化后）',
  `raw_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '剧本原始内容（用户粘贴的原文）',
  `total_episodes` int NULL DEFAULT 0 COMMENT '总集数',
  `story_synopsis` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '故事梗概',
  `characters_json` json NULL COMMENT '角色列表JSON',
  `source_type` int NULL DEFAULT 0 COMMENT '来源类型：0-手动创建 1-文件导入 2-AI生成',
  `parsing_status` int NULL DEFAULT 0 COMMENT '解析状态：0-未解析 1-解析中 2-解析完成 3-解析失败',
  `parsing_progress` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '解析进度描述',
  `summary` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI生成的剧本摘要',
  `genre` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '剧本类型/题材',
  `target_audience` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '目标受众',
  `duration_estimate` int NULL DEFAULT NULL COMMENT '预估总时长（分钟）',
  `scope` int NULL DEFAULT 3 COMMENT '可见范围：1-公开 2-私有 3-仅团队可见',
  `owner_type` int NULL DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NULL DEFAULT NULL COMMENT '拥有者ID',
  `ai_generated` tinyint NULL DEFAULT 0 COMMENT '是否由AI生成',
  `version` int NULL DEFAULT 0 COMMENT '乐观锁版本号',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_script_project`(`project_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '剧本表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_script
-- ----------------------------

-- ----------------------------
-- Table structure for afv_script_episode
-- ----------------------------
DROP TABLE IF EXISTS `afv_script_episode`;
CREATE TABLE `afv_script_episode`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `script_id` bigint NOT NULL COMMENT '所属剧本ID',
  `episode_number` int NULL DEFAULT NULL COMMENT '集号（从1开始）',
  `title` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '本集标题',
  `synopsis` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '本集剧情梗概',
  `raw_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '本集原始剧本内容',
  `duration_estimate` int NULL DEFAULT NULL COMMENT '预估时长（分钟）',
  `total_scenes` int NULL DEFAULT 0 COMMENT '本集总场次数',
  `source_type` int NULL DEFAULT 0 COMMENT '来源类型：0-AI解析 1-手动添加',
  `sort_order` int NULL DEFAULT 0 COMMENT '排列顺序',
  `parsing_status` int NULL DEFAULT 0 COMMENT '解析状态：0-未解析 1-解析中 2-解析完成 3-解析失败',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `version` int NULL DEFAULT 0 COMMENT '乐观锁版本号',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_episode_script`(`script_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '分集剧本表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_script_episode
-- ----------------------------

-- ----------------------------
-- Table structure for afv_script_scene_item
-- ----------------------------
DROP TABLE IF EXISTS `afv_script_scene_item`;
CREATE TABLE `afv_script_scene_item`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `episode_id` bigint NOT NULL COMMENT '所属分集ID',
  `script_id` bigint NOT NULL COMMENT '所属剧本ID',
  `scene_number` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场次编号（如1-1表示第1集第1场）',
  `scene_heading` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场景标头（如\"内景 客厅 夜\"）',
  `location` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场景地点',
  `time_of_day` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '时间段：日/夜/黄昏/清晨等',
  `int_ext` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '内外景标识：内景/外景/内外景',
  `characters` json NULL COMMENT '出场角色名列表JSON',
  `character_asset_ids` json NULL COMMENT '出场角色资产ID列表JSON',
  `scene_asset_id` bigint NULL DEFAULT NULL COMMENT '场景资产ID',
  `prop_asset_ids` json NULL COMMENT '道具资产ID列表JSON',
  `scene_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '场景氛围/环境描述',
  `dialogues` json NULL COMMENT '对白/动作元素列表JSON',
  `sort_order` int NULL DEFAULT 0 COMMENT '排列顺序',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `version` int NULL DEFAULT 0 COMMENT '乐观锁版本号',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_script_scene_episode`(`episode_id` ASC) USING BTREE,
  INDEX `idx_script_scene_script`(`script_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '剧本分场次表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_script_scene_item
-- ----------------------------

-- ----------------------------
-- Table structure for afv_storage_config
-- ----------------------------
DROP TABLE IF EXISTS `afv_storage_config`;
CREATE TABLE `afv_storage_config`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '配置名称',
  `type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '存储类型：local / aliyun_oss / tencent_cos / s3',
  `endpoint` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'OSS 端点地址',
  `bucket_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'OSS 存储桶名称',
  `access_key` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'OSS Access Key',
  `secret_key` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'OSS Secret Key',
  `region` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '区域',
  `base_path` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '存储根路径（本地为磁盘路径，OSS 为 key 前缀）',
  `custom_domain` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '自定义域名（CDN 域名等）',
  `is_default` tinyint NOT NULL DEFAULT 0 COMMENT '是否为默认存储配置',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `remark` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '备注',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '存储配置表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_storage_config
-- ----------------------------

-- ----------------------------
-- Table structure for afv_storyboard
-- ----------------------------
DROP TABLE IF EXISTS `afv_storyboard`;
CREATE TABLE `afv_storyboard`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `project_id` bigint NULL DEFAULT NULL COMMENT '所属项目ID',
  `script_id` bigint NULL DEFAULT NULL COMMENT '关联剧本ID',
  `title` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '分镜标题',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '分镜描述',
  `custom_columns` json NULL COMMENT '自定义列配置JSON',
  `scope` int NULL DEFAULT 3 COMMENT '可见范围：1-公开 2-私有 3-仅团队可见',
  `owner_type` int NULL DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NULL DEFAULT NULL COMMENT '拥有者ID',
  `total_duration` int NULL DEFAULT NULL COMMENT '预估总时长（秒）',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_storyboard_project`(`project_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '分镜脚本表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_storyboard
-- ----------------------------

-- ----------------------------
-- Table structure for afv_storyboard_episode
-- ----------------------------
DROP TABLE IF EXISTS `afv_storyboard_episode`;
CREATE TABLE `afv_storyboard_episode`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `storyboard_id` bigint NOT NULL COMMENT '所属分镜ID',
  `episode_number` int NULL DEFAULT NULL COMMENT '集号',
  `title` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '集标题',
  `synopsis` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '本集梗概',
  `sort_order` int NULL DEFAULT 0 COMMENT '排列顺序',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_sb_episode_storyboard`(`storyboard_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '分镜集表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_storyboard_episode
-- ----------------------------

-- ----------------------------
-- Table structure for afv_storyboard_item
-- ----------------------------
DROP TABLE IF EXISTS `afv_storyboard_item`;
CREATE TABLE `afv_storyboard_item`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `storyboard_id` bigint NOT NULL COMMENT '所属分镜ID',
  `storyboard_episode_id` bigint NULL DEFAULT NULL COMMENT '所属分镜集ID',
  `storyboard_scene_id` bigint NULL DEFAULT NULL COMMENT '所属分镜场次ID',
  `sort_order` int NULL DEFAULT 0 COMMENT '排列顺序',
  `shot_number` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '镜号',
  `auto_shot_number` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '自动编号（系统生成）',
  `image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '用户上传参考图片URL',
  `reference_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '外部参考图片URL',
  `video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '视频URL（最终成品）',
  `generated_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'AI生成的图片URL',
  `generated_video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'AI生成的视频URL',
  `shot_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '景别：远景/全景/中景/近景/特写',
  `duration` decimal(10, 2) NULL DEFAULT NULL COMMENT '预估时长（秒）',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '画面内容描述',
  `scene_expectation` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '画面期望描述（AI生图提示）',
  `sound` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '声音描述',
  `dialogue` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '台词/旁白',
  `sound_effect` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '音效',
  `music` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '配乐建议',
  `camera_movement` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '镜头运动：推/拉/摇/移/跟/升/降',
  `camera_angle` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '镜头角度：平视/俯视/仰视',
  `camera_equipment` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '摄像机装备',
  `focal_length` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '镜头焦段',
  `transition` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '转场效果：切/淡入/淡出/溶/划',
  `character_ids` json NULL COMMENT '出场角色子资产ID列表 JSON (List<Long> of AssetItem.id)',
  `scene_asset_item_id` bigint NULL DEFAULT NULL COMMENT '场景子资产ID (AssetItem.id)',
  `prop_ids` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '道具子资产ID列表 JSON (List<Long> of AssetItem.id)',
  `remark` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '备注',
  `custom_data` json NULL COMMENT '自定义扩展数据JSON',
  `ai_generated` tinyint NULL DEFAULT 0 COMMENT '是否由AI生成',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `video_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT 'AI生成视频时使用的提示词（保存以便复用和手动调整）',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_sb_item_storyboard`(`storyboard_id` ASC) USING BTREE,
  INDEX `idx_sb_item_scene`(`storyboard_scene_id` ASC) USING BTREE,
  INDEX `idx_sb_item_episode`(`storyboard_episode_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '分镜条目表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_storyboard_item
-- ----------------------------

-- ----------------------------
-- Table structure for afv_storyboard_scene
-- ----------------------------
DROP TABLE IF EXISTS `afv_storyboard_scene`;
CREATE TABLE `afv_storyboard_scene`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `episode_id` bigint NOT NULL COMMENT '所属分镜集ID',
  `storyboard_id` bigint NOT NULL COMMENT '所属分镜ID（冗余）',
  `scene_number` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场次编号',
  `scene_heading` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场景标头',
  `location` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场景地点',
  `time_of_day` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '时间段',
  `int_ext` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '内外景标识',
  `sort_order` int NULL DEFAULT 0 COMMENT '排列顺序',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-草稿 1-正常',
  `deleted` tinyint NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_sb_scene_episode`(`episode_id` ASC) USING BTREE,
  INDEX `idx_sb_scene_storyboard`(`storyboard_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '分镜场次表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_storyboard_scene
-- ----------------------------

-- ----------------------------
-- Table structure for afv_system_config
-- ----------------------------
DROP TABLE IF EXISTS `afv_system_config`;
CREATE TABLE `afv_system_config`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `config_key` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '配置键',
  `config_value` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '配置值',
  `remark` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '备注',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_config_key`(`config_key` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '系统配置表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_system_config
-- ----------------------------

-- ----------------------------
-- Table structure for afv_team
-- ----------------------------
DROP TABLE IF EXISTS `afv_team`;
CREATE TABLE `afv_team`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '团队名称',
  `logo` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '团队LOGO图片URL',
  `description` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '团队描述',
  `owner_user_id` bigint NOT NULL COMMENT '创建者用户ID',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '团队表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_team
-- ----------------------------

-- ----------------------------
-- Table structure for afv_team_member
-- ----------------------------
DROP TABLE IF EXISTS `afv_team_member`;
CREATE TABLE `afv_team_member`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `team_id` bigint NOT NULL COMMENT '所属团队ID',
  `user_id` bigint NOT NULL COMMENT '成员用户ID',
  `role` int NOT NULL DEFAULT 3 COMMENT '角色：1-创建者 2-管理员 3-普通成员',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `join_time` datetime NULL DEFAULT NULL COMMENT '加入时间',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_team_user`(`team_id` ASC, `user_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '团队成员表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_team_member
-- ----------------------------

-- ----------------------------
-- Table structure for afv_video_item
-- ----------------------------
DROP TABLE IF EXISTS `afv_video_item`;
CREATE TABLE `afv_video_item`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` bigint NOT NULL COMMENT '所属生视频任务ID',
  `platform_task_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '平台侧任务ID',
  `video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '生成的视频URL',
  `cover_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '视频封面图URL',
  `duration` int NULL DEFAULT NULL COMMENT '视频时长（秒）',
  `file_size` bigint NULL DEFAULT NULL COMMENT '文件大小（字节）',
  `status` int NULL DEFAULT 0 COMMENT '状态：0-生成中 1-成功 2-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '失败错误信息',
  `first_frame_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '视频首帧图片URL',
  `last_frame_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '视频尾帧图片URL',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_video_item_task`(`task_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '生视频条目表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_video_item
-- ----------------------------

-- ----------------------------
-- Table structure for afv_video_task
-- ----------------------------
DROP TABLE IF EXISTS `afv_video_task`;
CREATE TABLE `afv_video_task`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '任务唯一标识',
  `user_id` bigint NOT NULL COMMENT '发起用户ID',
  `project_id` bigint NULL DEFAULT NULL COMMENT '关联项目ID',
  `prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '生视频提示词',
  `prompt_template_id` bigint NULL DEFAULT NULL COMMENT '提示词模板ID',
  `generate_mode` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '生成模式：text2video/image2video',
  `first_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '首帧参考图片URL',
  `last_frame_image_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '尾帧参考图片URL',
  `reference_image_urls` json NULL COMMENT '参考图片URL列表JSON',
  `reference_video_urls` json NULL COMMENT '参考视频URL列表 JSON',
  `reference_audio_urls` json NULL COMMENT '参考音频URL列表 JSON',
  `ratio` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '画面比例（如16:9）',
  `resolution` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '分辨率（如1920x1080）',
  `duration` int NULL DEFAULT NULL COMMENT '视频时长（秒）',
  `watermark` tinyint NULL DEFAULT 0 COMMENT '是否添加水印',
  `generate_audio` tinyint NULL DEFAULT 0 COMMENT '是否生成配音',
  `seed` bigint NULL DEFAULT NULL COMMENT '随机种子（用于复现）',
  `camera_fixed` tinyint NULL DEFAULT 0 COMMENT '是否固定镜头',
  `count` int NULL DEFAULT 1 COMMENT '生成数量',
  `success_count` int NULL DEFAULT 0 COMMENT '已成功生成数量',
  `status` int NULL DEFAULT 0 COMMENT '任务状态：0-排队中 1-处理中 2-已完成 3-失败',
  `error_msg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '失败错误信息',
  `model_id` bigint NULL DEFAULT NULL COMMENT '使用的AI模型ID',
  `category` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '任务分类标签',
  `owner_type` int NULL DEFAULT NULL COMMENT '拥有者类型：1-个人 2-团队',
  `owner_id` bigint NULL DEFAULT NULL COMMENT '拥有者ID',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `task_id`(`task_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '生视频任务表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of afv_video_task
-- ----------------------------

-- ----------------------------
-- Table structure for sys_role
-- ----------------------------
DROP TABLE IF EXISTS `sys_role`;
CREATE TABLE `sys_role`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '角色名称',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '角色代码标识（如 admin、user）',
  `sort` int NULL DEFAULT 0 COMMENT '排列顺序',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `remark` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '备注说明',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `code`(`code` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '角色表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_role
-- ----------------------------
INSERT INTO `sys_role` VALUES (1, '超级管理员', 'admin', 1, 1, '系统超级管理员', 0, '2026-04-16 16:03:20', '2026-04-16 16:03:20');
INSERT INTO `sys_role` VALUES (2, '普通用户', 'user', 2, 1, '默认用户角色', 0, '2026-04-16 16:03:20', '2026-04-16 16:03:20');

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `username` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '登录用户名（唯一）',
  `password` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '登录密码（BCrypt加密）',
  `nickname` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '用户昵称',
  `avatar` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '头像URL',
  `email` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '邮箱地址',
  `phone` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '手机号码',
  `status` int NOT NULL DEFAULT 1 COMMENT '状态：0-禁用 1-启用',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '用户表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_user
-- ----------------------------

-- ----------------------------
-- Table structure for sys_user_role
-- ----------------------------
DROP TABLE IF EXISTS `sys_user_role`;
CREATE TABLE `sys_user_role`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `role_id` bigint NOT NULL COMMENT '角色ID',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除标志',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_user_role`(`user_id` ASC, `role_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '用户角色关联表' ROW_FORMAT = Dynamic;
