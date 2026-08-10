package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * Agent 消息实体
 * <p>
 * 对应数据库表：afv_agent_message
 * 存储 AI Agent 对话中的每条消息记录。
 */
@TableName("afv_agent_message")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentMessage extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属对话ID（UUID），关联 AgentConversation.conversationId */
    private String conversationId;

    /** 产生该消息的运行 ID */
    private String runId;

    /** 事件投影幂等键 */
    private String projectionKey;

    /** 消息角色：user-用户 assistant-AI system-系统 tool-工具 */
    private String role;

    /** 消息文本内容 */
    private String content;

    /** 引用资源 JSON，存储消息中引用的文件/图片等 */
    private String referencesJson;

    /** 工具调用名称（当 role=tool 时） */
    private String toolName;

    /** 工具执行状态：running/success/error/cancelled/rejected */
    private String toolStatus;

    /** 工具调用 ID（关联同一次工具调用的发起和结果） */
    private String toolCallId;

    /** 父级工具调用 ID（子 Agent 事件归属到父工具调用） */
    private String parentToolCallId;

    /** AI推理过程内容（思维链） */
    private String reasoningContent;

    /** AI推理耗时（毫秒） */
    private Long reasoningDurationMs;

    /** 消息在对话中的排列顺序 */
    private Long messageOrder;
}
