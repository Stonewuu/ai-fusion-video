package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

import java.time.LocalDateTime;

/**
 * Agent 对话索引实体
 * <p>
 * 对应数据库表：afv_agent_conversation
 * 记录用户与 AI Agent 之间的对话会话元数据。
 */
@TableName("afv_agent_conversation")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentConversation extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 对话唯一标识（UUID），用于前后端通信 */
    private String conversationId;

    /** 所属用户ID */
    private Long userId;

    /** 关联项目ID */
    private Long projectId;

    /** 上下文类型，如 project/script/storyboard */
    private String contextType;

    /** Agent 类型，如 script_parser/storyboard_creator */
    private String agentType;

    /** 对话分类标签 */
    private String category;

    /** 上下文对象ID，关联具体的业务实体 */
    private Long contextId;

    /** 对话标题 */
    @Builder.Default
    private String title = "新对话";

    /** 消息总数 */
    @Builder.Default
    private Integer messageCount = 0;

    /** 下一条消息的全历史唯一顺序号 */
    @Builder.Default
    private Long nextMessageOrder = 1L;

    /** 最后消息时间 */
    private LocalDateTime lastMessageTime;

    /** 对话状态：active-进行中 closed-已关闭 */
    private String status;
}
