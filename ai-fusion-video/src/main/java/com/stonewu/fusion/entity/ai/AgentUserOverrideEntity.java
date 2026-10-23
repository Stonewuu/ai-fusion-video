package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_agent_user_override")
public class AgentUserOverrideEntity {
    /**
     * 所属用户
     */
    private Long userId;

    /**
     * 内置智能体标识
     */
    private String agentType;

    /**
     * 用户覆盖的配置字段，缺省字段继承内置值
     */
    private String overridesJson;

    /**
     * 配置修订号，用于并发编辑校验
     */
    private Long revision;

    /**
     * 创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 更新时间
     */
    private LocalDateTime updatedAt;
}
