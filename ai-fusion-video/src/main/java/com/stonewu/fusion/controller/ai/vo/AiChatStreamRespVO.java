package com.stonewu.fusion.controller.ai.vo;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.entity.ai.AgentMessage;
import lombok.Data;
import lombok.experimental.Accessors;

import java.time.Instant;
import java.util.List;

/**
 * AI 助手流式响应 VO（SSE 事件数据）
 */
@Data
@Accessors(chain = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class AiChatStreamRespVO {

    /**
     * Durable wire schema version.
     */
    private Integer schemaVersion;

    /**
     * Authoritative Agent run ID.
     */
    private String runId;

    /** 产生消息的运行；展示流可以同时包含多个子运行。 */
    private String sourceRunId;

    /** 审批等控制操作对应的运行。 */
    private String controlRunId;

    /**
     * Monotonic sequence within runId.
     */
    private Long sequence;

    /**
     * 消息 ID
     */
    private String messageId;

    /**
     * 会话 ID
     */
    private String conversationId;

    /**
     * 输出类型：REASONING / CONTENT / TOOL_CALL_STARTED / TOOL_CALL / TOOL_FINISHED / SUB_AGENT_FINISHED / DONE / ERROR / CANCELLED
     */
    private String outputType;

    /** 重连快照中的完整消息来自 MySQL，草稿和后续增量来自 Redis。 */
    private List<AgentMessage> historyMessages;
    private List<AiChatStreamRespVO> pendingEvents;
    private String runStatus;

    /**
     * 文本内容（增量）
     */
    private String content;

    /** 完整消息替换草稿时为 replace，增量事件不设置。 */
    private String contentMode;

    /**
     * 思考内容（增量）
     */
    private String reasoningContent;

    /**
     * 思考开始时间
     */
    private Long reasoningStartTime;

    /**
     * 思考耗时（毫秒，首个 CONTENT 事件携带）
     */
    private Long reasoningDurationMs;

    /**
     * 工具调用信息
     */
    private List<ToolCallVO> toolCalls;

    /**
     * 工具调用 ID
     */
    private String toolCallId;

    /**
     * 工具名
     */
    private String toolName;

    /**
     * 工具执行结果
     */
    private String toolResult;

    /**
     * 工具执行状态：success / error
     */
    private String toolStatus;

    /**
     * 父级工具调用 ID（子 Agent 输出时用于归属映射）
     */
    private String parentToolCallId;

    /**
     * 产生此事件的 Agent 名称
     */
    private String agentName;

    /**
     * 是否结束
     */
    private Boolean finished;

    /**
     * 错误信息
     */
    private String error;

    /**
     * Sanitized AgentScope event source.
     */
    private String source;

    /**
     * AgentScope reply identity when present.
     */
    private String replyId;

    /**
     * AgentScope block identity when present.
     */
    private String blockId;

    /**
     * Stable raw event identity.
     */
    private String rawEventId;

    /**
     * Exhaustively mapped raw event discriminator.
     */
    private String rawEventType;

    /**
     * Original event creation time.
     */
    private Instant createdAt;

    /**
     * Platform control discriminator, currently USER_CONFIRM_REQUIRED.
     */
    private String controlType;

    /**
     * Server-persisted tool set for an actionable confirmation.
     */
    private List<PendingToolCallVO> pendingToolCalls;

    /**
     * Exact user decisions for the persisted confirmation tool set.
     */
    private List<ToolConfirmationDecisionVO> decisions;

    /**
     * Server-authoritative action expiry.
     */
    private Instant expiresAt;

    /**
     * Machine-readable reason for a terminal cancellation.
     */
    private String cancellationReason;

    @Data
    @Accessors(chain = true)
    public static class ToolCallVO {
        private String id;
        private String name;
        private String arguments;
    }

    @Data
    @Accessors(chain = true)
    public static class PendingToolCallVO {
        private String toolCallId;
        private String toolName;
        private String argumentsPreview;
        private JsonNode review;
    }

    @Data
    @Accessors(chain = true)
    public static class ToolConfirmationDecisionVO {
        private String toolCallId;
        private Boolean approved;
    }
}
