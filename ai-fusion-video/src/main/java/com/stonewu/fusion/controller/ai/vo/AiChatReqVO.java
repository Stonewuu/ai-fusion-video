package com.stonewu.fusion.controller.ai.vo;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentContext;
import lombok.Data;
import lombok.experimental.Accessors;

import java.util.List;
import java.util.Map;

/**
 * AI 助手对话请求 VO
 */
@Data
@Accessors(chain = true)
public class AiChatReqVO {

    /**
     * 用户消息
     */
    private String message;

    /**
     * 会话ID（为空则新建）
     */
    private String conversationId;

    /**
     * AI 模型 ID
     */
    private Long modelId;

    /**
     * 本次请求使用的思考等级。
     */
    private String reasoningEffort;

    /**
     * Agent 类型
     */
    private String agentType;

    /**
     * 会话分类
     */
    private String category;

    /**
     * 自定义会话标题（不传则使用消息前50字）
     */
    private String title;

    /**
     * 项目 ID
     */
    private Long projectId;

    /**
     * 当前画布的定位上下文，受理时由服务端重新授权和固定。
     */
    private CanvasAgentContext canvasContext;

    /**
     * 上下文引用
     */
    private Map<String, Object> context;

    /**
     * 引用信息 JSON（纯存储，透传给前端）
     */
    private String referencesJson;

    /**
     * 系统提示词（可选覆盖）
     */
    private String systemPrompt;

    /**
     * 详细指令（可选覆盖）
     */
    private String instruction;

    /**
     * 前端指定的启用工具列表（与 Agent 白名单取交集）
     */
    private List<String> enabledTools;

    /**
     * 用户在输入区显式激活的 Skill 名称。
     */
    private List<String> enabledSkills;

    /**
     * false 表示本次仅使用显式 Skill；默认沿用智能体默认 Skill 并追加本次选择。
     */
    private Boolean inheritAgentSkills;

    /**
     * 受理链路内部固定的用户配置；HTTP 请求不能提供或替换此快照。
     */
    @JsonIgnore
    private String agentConfigurationSnapshot;

    /**
     * 画布预览时固定的 Skill 正文，仅由服务端受理链路设置。
     */
    @JsonIgnore
    private String agentSkillSnapshot;

    /**
     * 用户在输入区主动引用的 MCP 工具；仅筛选 MCP，不影响平台工具。
     */
    private List<String> enabledMcpTools;

    /**
     * 用户随本轮消息提交的图片、视频、音频或文件输入。
     */
    private List<AiMultimodalInputVO> multimodalInputs;

    /**
     * 自动引用（当前页面上下文，包含 type + id）
     */
    private List<AiReferenceVO> autoReferences;

    /**
     * 用户手动添加的引用
     */
    private List<AiReferenceVO> references;

    /**
     * 是否启用并行工具执行（Multi-Agent 模式）
     */
    private Boolean enableParallelTools;

    /**
     * Tool execution policy: DEFAULT / ALWAYS_ASK / ALWAYS_ALLOW / FULL_ACCESS.
     */
    private String toolExecutionMode;
}
