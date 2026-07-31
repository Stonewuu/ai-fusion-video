package com.stonewu.fusion.config.ai;

import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * AI Agent 注册表
 * <p>
 * 在代码中静态定义所有 Agent 配置，提示词从 resources/prompts/agents/ 目录加载。
 * 新增 Agent 只需在 {@link #registerBuiltinAgents()} 中添加。
 */
@Component
public class AiAgentRegistry {

        private static final String SUB_AGENT_MESSAGE_SCHEMA = """
                        {
                          "type": "object",
                          "properties": {
                            "message": {
                              "type": "string",
                              "description": "发送给子 Agent 的任务消息"
                            }
                          },
                          "required": ["message"],
                          "additionalProperties": false
                        }
                        """;

        private final Map<String, AiAgentDefinition> agentMap = new LinkedHashMap<>();

        public AiAgentRegistry() {
                registerBuiltinAgents();
        }

        /**
         * 从 classpath 加载提示词文件
         *
         * @param fileName 文件名（相对于 prompts/agents/ 目录）
         * @return 提示词内容
         */
        private static String loadPrompt(String fileName) {
                try {
                        ClassPathResource resource = new ClassPathResource("prompts/agents/" + fileName);
                        try (InputStream is = resource.getInputStream()) {
                                return new String(is.readAllBytes(), StandardCharsets.UTF_8).trim();
                        }
                } catch (IOException e) {
                        throw new RuntimeException("加载提示词文件失败: prompts/agents/" + fileName, e);
                }
        }

        /**
         * 注册内置 Agent 定义
         */
        private void registerBuiltinAgents() {
                registerAiMediaAgent();
                registerConceptVisualizerAgent();
                registerScriptFullParseAgent();
                registerScriptStoryToScriptAgent();
                registerScriptEpisodeUploadAgent();
                registerScriptAssistantAgent();
                registerScriptToStoryboardAgent();
                registerEpisodeSceneWriterAgent();
                registerEpisodeScriptCreatorAgent();
                registerEpisodeStoryboardWriterAgent();
                registerStoryboardAssetPreprocessorAgent();
                registerAssetImageGenerationAgent();
                registerAssetImageExecutorAgent();
                registerStoryboardFrameGenAgent();
                registerStoryboardFrameExecutorAgent();
                registerStoryboardVideoGenAgent();
                registerStoryboardVideoExecutorAgent();
        }

        // ========== 各 Agent 定义 ==========

        private void registerAiMediaAgent() {
                register(AiAgentDefinition.builder()
                                .type("ai_media")
                                .name("默认助手")
                                .toolNames(List.of(
                                                "list_my_projects", "get_project", "save_project", "delete_project",
                                                "get_project_script", "list_project_assets", "get_project_storyboard",
                                                "get_script", "get_script_structure", "get_script_episode",
                                                "update_script", "update_script_info", "save_script_episode",
                                                "save_script_scene_items", "update_script_scene",
                                                "manage_script_scenes", "delete_script_child_resource",
                                                "get_asset", "create_asset", "batch_create_assets", "update_asset",
                                                "add_asset_item", "update_asset_item", "delete_asset_resource",
                                                "get_storyboard", "insert_storyboard_item",
                                                "update_storyboard_scene", "update_storyboard_item",
                                                "delete_storyboard_child_resource",
                                                "get_generation_model_capabilities",
                                                "generate_image",
                                                "query_asset_metadata"))
                                .systemPrompt(loadPrompt("ai-media.system.md"))
                                .instructionTemplate(loadPrompt("ai-media.instruction.md"))
                                .enableTools(1)
                                .build());
        }

        private void registerConceptVisualizerAgent() {
                register(AiAgentDefinition.builder()
                                .type("concept_visualizer")
                                .name("概念可视化")
                                .systemPrompt(loadPrompt("concept-visualizer.system.md"))
                                .instructionTemplate("")
                                .enableTools(0)
                                .build());
        }

        private void registerScriptFullParseAgent() {
                register(AiAgentDefinition.builder()
                                .type("script_full_parse")
                                .name("完整剧本解析")
                                .toolNames(List.of(
                                                "get_project_script", "list_project_assets", "batch_create_assets",
                                                "update_script_info", "save_script_episode",
                                                "get_script_structure", "query_asset_metadata"))
                                .subAgentTools(List.of(
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("episode_scene_writer")
                                                                .displayName("分集场次编写器")
                                                                .description("对指定分集内容进行场次拆解和编写。输入分集编号和内容摘要，子Agent会自动解析并保存场次数据。可同时调用多个实例并行处理不同分集。调用时只传声明中要求的业务参数，不要传 session_id，框架会自动维护会话。")
                                                                .parametersSchema(
                                                                                """
                                                                                                {
                                                                                                  "type": "object",
                                                                                                  "properties": {
                                                                                                    "scriptEpisodeId": {
                                                                                                      "type": "integer",
                                                                                                      "description": "剧本分集记录ID（从 save_script_episode 的返回值中获取）"
                                                                                                    }
                                                                                                  },
                                                                                                  "additionalProperties": false,
                                                                                                  "required": ["scriptEpisodeId"]
                                                                                                }""")
                                                                .refAgentType("episode_scene_writer")
                                                                .outputSchema("""
                                                                                {
                                                                                  "type": "object",
                                                                                  "description": "子 Agent 的场次解析输出",
                                                                                  "properties": {
                                                                                    "scriptEpisodeId": { "type": "integer", "description": "处理的剧本分集ID" },
                                                                                    "sceneCount": { "type": "integer", "description": "解析出的场次数量" },
                                                                                    "status": { "type": "string", "enum": ["success", "partial", "failed"], "description": "处理状态" },
                                                                                    "message": { "type": "string", "description": "处理结果描述" }
                                                                                  },
                                                                                  "required": ["scriptEpisodeId", "sceneCount", "status"]
                                                                                }""")
                                                                .systemPromptOverride(loadPrompt(
                                                                                "script-full-parse_episode-scene-writer.override.md"))
                                                                .build()))
                                .systemPrompt(loadPrompt("script-full-parse.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <script_id>{scriptId}</script_id>
                                                </task_context>""")
                                .defaultUserMessage("请解析项目 {projectId} 的剧本（ID: {scriptId}），将原文内容解析为结构化的分集、场次和对白数据。")
                                .enableTools(1)
                                .build());
        }

        private void registerScriptStoryToScriptAgent() {
                register(AiAgentDefinition.builder()
                                .type("story_to_script")
                                .name("故事转剧本")
                                .toolNames(List.of(
                                                "list_project_assets", "batch_create_assets",
                                                "update_script_info", "save_script_episode",
                                                "get_project_script", "query_asset_metadata", "get_script_structure"))
                                .subAgentTools(List.of(
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("episode_script_creator")
                                                                .displayName("分集剧本创作器")
                                                                .description("对指定分集进行对白和场次创作。输入分集编号，子Agent会自动查询大纲、设计场次、创作对白并保存。可同时调用多个实例并行处理不同分集。调用时只传声明中要求的业务参数，不要传 session_id，框架会自动维护会话。")
                                                                .parametersSchema(
                                                                                """
                                                                                                {
                                                                                                  "type": "object",
                                                                                                  "properties": {
                                                                                                    "scriptEpisodeId": {
                                                                                                      "type": "integer",
                                                                                                      "description": "剧本分集记录ID（从 save_script_episode 的返回值中获取）"
                                                                                                    }
                                                                                                  },
                                                                                                  "additionalProperties": false,
                                                                                                  "required": ["scriptEpisodeId"]
                                                                                                }""")
                                                                .refAgentType("episode_script_creator")
                                                                .outputSchema("""
                                                                                {
                                                                                  "type": "object",
                                                                                  "description": "子 Agent 的场次创作输出",
                                                                                  "properties": {
                                                                                    "scriptEpisodeId": { "type": "integer", "description": "处理的剧本分集ID" },
                                                                                    "sceneCount": { "type": "integer", "description": "创作的场次数量" },
                                                                                    "status": { "type": "string", "enum": ["success", "partial", "failed"], "description": "处理状态" },
                                                                                    "message": { "type": "string", "description": "处理结果描述" }
                                                                                  },
                                                                                  "required": ["scriptEpisodeId", "sceneCount", "status"]
                                                                                }""")
                                                                .systemPromptOverride(loadPrompt(
                                                                                "script-story-to-script_episode-script-creator.override.md"))
                                                                .build()))
                                .systemPrompt(loadPrompt("script-story-to-script.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <script_id>{scriptId}</script_id>
                                                </task_context>""")
                                .defaultUserMessage("请根据项目 {projectId} 的剧本大纲（ID: {scriptId}），创作完整的分集场次和对白内容。")
                                .enableTools(1)
                                .build());
        }

        private void registerScriptEpisodeUploadAgent() {
                register(AiAgentDefinition.builder()
                                .type("script_episode_parse")
                                .name("分集上传解析")
                                .toolNames(List.of(
                                                "list_project_assets", "batch_create_assets",
                                                "update_script_info", "save_script_scene_items",
                                                "get_script_episode", "get_script_structure",
                                                "manage_script_scenes", "query_asset_metadata", "get_project"))
                                .systemPrompt(loadPrompt("script-episode-upload.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <script_id>{scriptId}</script_id>
                                                <episode_id>{episodeId}</episode_id>
                                                </task_context>""")
                                .defaultUserMessage("请解析分集（episodeId: {episodeId}）的剧本原文，将内容解析为结构化的场次和对白数据。")
                                .enableTools(1)
                                .build());
        }

        private void registerScriptAssistantAgent() {
                register(AiAgentDefinition.builder()
                                .type("script_assistant")
                                .name("剧本对话助手")
                                .toolNames(List.of(
                                                "get_script_structure", "get_script_episode",
                                                "update_script_scene", "manage_script_scenes",
                                                "list_project_assets", "create_asset"))
                                .systemPrompt(loadPrompt("script-assistant.system.md"))
                                .instructionTemplate("")
                                .enableTools(1)
                                .build());
        }

        private void registerScriptToStoryboardAgent() {
                register(AiAgentDefinition.builder()
                                .type("script_to_storyboard")
                                .name("剧本转分镜")
                                .toolNames(List.of(
                                                "get_project", "get_script_structure",
                                                "list_project_assets", "get_storyboard",
                                                "get_project_storyboard"))
                                .subAgentTools(List.of(
                                                // 子资产预处理子 Agent（串行，先执行）
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("storyboard_asset_preprocessor")
                                                                .displayName("子资产预处理器")
                                                                .description("分析所有分集剧本内容，识别角色/场景/道具的外观变化，统一创建所需的子资产变体并保存到数据库。此工具必须在 episode_storyboard_writer 之前调用，且只调用一次。调用时只传声明中要求的业务参数，不要传 session_id，框架会自动维护会话。")
                                                                .parametersSchema(
                                                                                """
                                                                                                {
                                                                                                  "type": "object",
                                                                                                  "properties": {
                                                                                                    "scriptEpisodeIds": {
                                                                                                      "type": "string",
                                                                                                      "description": "所有需要处理的剧本分集ID列表，逗号分隔，如 '1,2,3'"
                                                                                                    }
                                                                                                  },
                                                                                                  "additionalProperties": false,
                                                                                                  "required": ["scriptEpisodeIds"]
                                                                                                }""")
                                                                .refAgentType("storyboard_asset_preprocessor")
                                                                .outputSchema("""
                                                                                {
                                                                                  "type": "object",
                                                                                  "description": "子资产预处理输出",
                                                                                  "properties": {
                                                                                    "createdCount": { "type": "integer", "description": "新创建的子资产数量" },
                                                                                    "status": { "type": "string", "enum": ["success", "failed"], "description": "处理状态" },
                                                                                    "message": { "type": "string", "description": "处理结果描述" }
                                                                                  },
                                                                                  "required": ["status"]
                                                                                }""")
                                                                .systemPromptOverride(loadPrompt(
                                                                                "script-to-storyboard_asset-preprocessor.override.md"))
                                                                .build(),
                                                // 分集分镜编写子 Agent（并行，后执行）
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("episode_storyboard_writer")
                                                                .displayName("分集分镜编写器")
                                                                .description("对指定分集进行分镜转换。输入分集编号，子Agent会自动查询场次内容、获取最新资产列表（含预处理器已创建的子资产）、设计镜头并保存分镜数据。可同时调用多个实例并行处理不同分集。调用时只传声明中要求的业务参数，不要传 session_id，框架会自动维护会话。")
                                                                .parametersSchema(
                                                                                """
                                                                                                {
                                                                                                  "type": "object",
                                                                                                  "properties": {
                                                                                                    "scriptEpisodeId": {
                                                                                                      "type": "integer",
                                                                                                      "description": "剧本分集记录ID（从 get_script_structure 获取）"
                                                                                                    }
                                                                                                  },
                                                                                                  "additionalProperties": false,
                                                                                                  "required": ["scriptEpisodeId"]
                                                                                                }""")
                                                                .refAgentType("episode_storyboard_writer")
                                                                .outputSchema("""
                                                                                {
                                                                                  "type": "object",
                                                                                  "description": "子 Agent 的分镜转换输出",
                                                                                  "properties": {
                                                                                    "scriptEpisodeId": { "type": "integer", "description": "处理的剧本分集ID" },
                                                                                    "shotCount": { "type": "integer", "description": "生成的镜头总数" },
                                                                                    "sceneCount": { "type": "integer", "description": "处理的场次数量" },
                                                                                    "status": { "type": "string", "enum": ["success", "partial", "failed"], "description": "处理状态" },
                                                                                    "message": { "type": "string", "description": "处理结果描述" }
                                                                                  },
                                                                                  "required": ["scriptEpisodeId", "shotCount", "sceneCount", "status"]
                                                                                }""")
                                                                .systemPromptOverride(loadPrompt(
                                                                                "script-to-storyboard_episode-storyboard-writer.override.md"))
                                                                .build()))
                                .systemPrompt(loadPrompt("script-to-storyboard.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <storyboard_id>{storyboardId}</storyboard_id>
                                                <script_id>{scriptId}</script_id>
                                                </task_context>""")
                                .defaultUserMessage("请根据项目 {projectId} 的剧本（ID: {scriptId}），将结构化剧本数据转换为分镜表。")
                                .enableTools(1)
                                .build());
        }

        private void registerEpisodeSceneWriterAgent() {
                register(AiAgentDefinition.builder()
                                .type("episode_scene_writer")
                                .name("分集场次编写器")
                                .toolNames(List.of(
                                                "get_script_episode", "list_project_assets",
                                                "save_script_scene_items", "get_project_script",
                                                "get_script_scene"))
                                .systemPrompt(loadPrompt("episode-scene-writer.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <script_id>{scriptId}</script_id>
                                                </task_context>

                                                请根据主 Agent 提供的 episodeId，查询该集原文并解析场次数据。""")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分集剧本创作子 Agent
         * <p>
         * 每个实例只处理一集，自主完成查询大纲→获取资产→创作场次对白→保存全流程。
         */
        private void registerEpisodeScriptCreatorAgent() {
                register(AiAgentDefinition.builder()
                                .type("episode_script_creator")
                                .name("分集剧本创作器")
                                .toolNames(List.of(
                                                "get_script_episode", "list_project_assets", "save_script_scene_items"))
                                .systemPrompt(loadPrompt("episode-script-creator.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <script_id>{scriptId}</script_id>
                                                </task_context>

                                                请根据主 Agent 提供的 episodeId，查询该集大纲并创作完整的场次和对白。""")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分集分镜编写子 Agent
         * <p>
         * 每个实例只处理一集，自主完成查场次→设计镜头→匹配子资产→保存分镜全流程。
         */
        private void registerEpisodeStoryboardWriterAgent() {
                register(AiAgentDefinition.builder()
                                .type("episode_storyboard_writer")
                                .name("分集分镜编写器")
                                .toolNames(List.of(
                                                "get_script_episode", "get_script_scene", "list_project_assets",
                                                "save_storyboard_episode", "save_storyboard_scene_shots"))
                                .systemPrompt(loadPrompt("episode-storyboard-writer.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <storyboard_id>{storyboardId}</storyboard_id>
                                                <script_id>{scriptId}</script_id>
                                                <script_episode_id>{scriptEpisodeId}</script_episode_id>
                                                </task_context>

                                                请根据任务上下文中的 scriptEpisodeId，查询该集剧本内容并设计分镜。""")
                                .defaultUserMessage("请为剧本分集（scriptEpisodeId: {scriptEpisodeId}）生成分镜，并保存到分镜脚本 {storyboardId}。")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分镜子资产预处理子 Agent
         * <p>
         * 串行执行（只调用一次），负责分析所有分集剧本内容，
         * 识别角色/场景/道具的外观变化，统一创建子资产变体，
         * 返回完整的 assetItemMapping 供分镜编写子 Agent 使用。
         */
        private void registerStoryboardAssetPreprocessorAgent() {
                register(AiAgentDefinition.builder()
                                .type("storyboard_asset_preprocessor")
                                .name("子资产预处理器")
                                .toolNames(List.of(
                                                "get_script_episode", "list_project_assets",
                                                "query_asset_items", "batch_create_asset_items",
                                                "query_asset_metadata"))
                                .systemPrompt(loadPrompt("storyboard-asset-preprocessor.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <script_id>{scriptId}</script_id>
                                                </task_context>

                                                请根据主 Agent 提供的 episodeIds，逐集分析剧本内容并创建所需的子资产变体。""")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册资产图片自动生成主 Agent
         * <p>
         * 负责获取项目画风、收集无图子资产、按子资产维度并行分发子 Agent。
         */
        private void registerAssetImageGenerationAgent() {
                register(AiAgentDefinition.builder()
                                .type("asset_image_gen")
                                .name("资产图片自动生成")
                                .toolNames(List.of(
                                                "get_project", "query_asset_items", "list_project_assets"))
                                .subAgentTools(List.of(
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("generate_asset_image")
                                                                .displayName("为子资产生成图片")
                                                                .description("""
                                                                                为单个子资产生成AI图片并自动保存。每次调用只处理一个子资产，可在同一轮同时调用多个实例并行执行。

                                                                                调用时 message 必须包含以下信息（每行一个键值对）：
                                                                                - assetId: 主资产ID（数字，必传）
                                                                                - itemId: 子资产ID（数字，必传）
                                                                                - projectId: 项目ID（数字，必传）
                                                                                - 不要额外传 session_id，框架会自动维护会话

                                                                                message 格式示例：
                                                                                请为子资产生成图片。
                                                                                assetId: 1
                                                                                itemId: 3
                                                                                projectId: 5""")
                                                                .parametersSchema(SUB_AGENT_MESSAGE_SCHEMA)
                                                                .refAgentType("asset_image_executor")
                                                                .build()))
                                .systemPrompt(loadPrompt("asset-image-generation.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                </task_context>""")
                                .defaultUserMessage("请为项目 {projectId} 的资产自动生成图片。")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册资产图片生成执行子 Agent
         * <p>
         * 每个实例只处理一个子资产，自主完成查资产→编排prompt→生图→回填全流程。
         */
        private void registerAssetImageExecutorAgent() {
                register(AiAgentDefinition.builder()
                                .type("asset_image_executor")
                                .name("资产图片生成执行器")
                                .toolNames(List.of(
                                                "get_project", "query_asset_items", "get_generation_model_capabilities", "generate_image",
                                                "update_asset_image"))
                                .systemPrompt(loadPrompt("asset-image-executor.system.md"))
                                .instructionTemplate("")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分镜首尾帧生成主 Agent
         * <p>
         * 负责读取前端指定的镜头、帧类型和提示词，并按镜头维度分发子 Agent。
         */
        private void registerStoryboardFrameGenAgent() {
                register(AiAgentDefinition.builder()
                                .type("storyboard_frame_gen")
                                .name("分镜首尾帧生成")
                                .toolNames(List.of(
                                                "get_project", "get_storyboard", "get_storyboard_scene_items"))
                                .subAgentTools(List.of(
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("generate_storyboard_frame")
                                                                .displayName("为镜头生成首尾帧")
                                                                .description("""
                                                                                为单个分镜镜头生成首帧或尾帧图片并自动保存。每次调用只处理一个镜头的一个帧，可在同一轮同时调用多个实例并行执行。

                                                                                调用时 message 必须包含以下信息（每行一个键值对）：
                                                                                - storyboardItemId: 分镜条目ID（数字，必传）
                                                                                - projectId: 项目ID（数字，必传）
                                                                                - frameType: 帧类型，first 或 last（必传）
                                                                                - framePrompt: 用户确认后的首尾帧提示词（必传）
                                                                                - 不要额外传 session_id，框架会自动维护会话

                                                                                message 格式示例：
                                                                                请为分镜镜头生成首尾帧。
                                                                                storyboardItemId: 42
                                                                                projectId: 5
                                                                                frameType: first
                                                                                framePrompt: 生成该镜头视频的首帧定格图……""")
                                                                .parametersSchema(SUB_AGENT_MESSAGE_SCHEMA)
                                                                .refAgentType("storyboard_frame_executor")
                                                                .build()))
                                .systemPrompt(loadPrompt("storyboard-frame-gen.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <storyboard_id>{storyboardId}</storyboard_id>
                                                <frame_type>{frameType}</frame_type>
                                                <frame_prompt>{framePrompt}</frame_prompt>
                                                </task_context>""")
                                .defaultUserMessage("请为项目 {projectId} 的分镜镜头生成首尾帧。")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分镜首尾帧生成执行子 Agent
         * <p>
         * 每个实例只处理一个镜头的一个帧，自主完成查上下文、生图、回填全流程。
         */
        private void registerStoryboardFrameExecutorAgent() {
                register(AiAgentDefinition.builder()
                                .type("storyboard_frame_executor")
                                .name("分镜首尾帧生成执行器")
                                .toolNames(List.of(
                                                "get_project", "get_storyboard_scene_items",
                                                "get_generation_model_capabilities", "generate_image",
                                                "update_storyboard_item_frame"))
                                .systemPrompt(loadPrompt("storyboard-frame-executor.system.md"))
                                .instructionTemplate("")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分镜视频生成主 Agent
         * <p>
         * 负责获取项目信息、分析选中的分镜镜头、按镜头维度并行分发子 Agent。
         */
        private void registerStoryboardVideoGenAgent() {
                register(AiAgentDefinition.builder()
                                .type("storyboard_video_gen")
                                .name("分镜视频生成")
                                .toolNames(List.of(
                                                "get_project", "get_storyboard", "get_storyboard_scene_items"))
                                .subAgentTools(List.of(
                                                AiAgentDefinition.SubAgentToolDef.builder()
                                                                .toolName("generate_storyboard_video")
                                                                .displayName("为镜头生成视频")
                                                                .description("""
                                                                                为单个分镜镜头生成AI视频并自动保存。每次调用只处理一个镜头，可在同一轮同时调用多个实例并行执行。

                                                                                调用时 message 必须包含以下信息（每行一个键值对）：
                                                                                - storyboardItemId: 分镜条目ID（数字，必传）
                                                                                - projectId: 项目ID（数字，必传）
                                                                                - 不要额外传 session_id，框架会自动维护会话

                                                                                message 格式示例：
                                                                                请为分镜镜头生成视频。
                                                                                storyboardItemId: 42
                                                                                projectId: 5""")
                                                                .parametersSchema(SUB_AGENT_MESSAGE_SCHEMA)
                                                                .refAgentType("storyboard_video_executor")
                                                                .build()))
                                .systemPrompt(loadPrompt("storyboard-video-gen.system.md"))
                                .instructionTemplate("""
                                                <task_context>
                                                <project_id>{projectId}</project_id>
                                                <storyboard_id>{storyboardId}</storyboard_id>
                                                </task_context>""")
                                .defaultUserMessage("请为项目 {projectId} 的分镜镜头生成视频。")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册分镜视频生成执行子 Agent
         * <p>
         * 每个实例只处理一个镜头，自主完成查上下文→编排prompt→生视频→回填全流程。
         */
        private void registerStoryboardVideoExecutorAgent() {
                register(AiAgentDefinition.builder()
                                .type("storyboard_video_executor")
                                .name("分镜视频生成执行器")
                                .toolNames(List.of(
                                                "get_project", "get_storyboard_scene_items",
                                                "get_generation_model_capabilities", "generate_video", "update_storyboard_item_video"))
                                .systemPrompt(loadPrompt("storyboard-video-executor.system.md"))
                                .instructionTemplate("")
                                .enableTools(1)
                                .build());
        }

        /**
         * 注册 Agent 定义
         */
        public void register(AiAgentDefinition definition) {
                agentMap.put(definition.getType(), definition);
        }

        /**
         * 根据类型获取 Agent 定义
         *
         * @param type Agent 类型
         * @return Agent 定义，不存在返回 null
         */
        public AiAgentDefinition getByType(String type) {
                return agentMap.get(type);
        }

        /**
         * 获取所有 Agent 定义
         */
        public List<AiAgentDefinition> getAll() {
                return new ArrayList<>(agentMap.values());
        }
}
