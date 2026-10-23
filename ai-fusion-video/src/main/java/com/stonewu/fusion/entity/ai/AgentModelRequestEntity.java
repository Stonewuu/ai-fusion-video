package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_agent_model_request")
public class AgentModelRequestEntity {
    /**
     * 模型调用标识，同时作为用量账本调用标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 实际执行智能体任务标识
     */
    private String runId;

    /**
     * 冻结的模型编号
     */
    private Long modelId;

    /**
     * 发起原调用的工作者标识
     */
    private String requestOwner;

    /**
     * 发起原调用的租约代次
     */
    private Long requestEpoch;

    /**
     * 模型调用状态
     */
    private String status;

    /**
     * 是否仍占用模型并发额度
     */
    private Boolean permitHeld;

    /**
     * 原调用完整响应接收时间（UTC）
     */
    private LocalDateTime responseReceivedAt;

    /**
     * 核实版本
     */
    private Long revision;

    /**
     * 发送意图落库时间（UTC）
     */
    private LocalDateTime createdAt;

    /**
     * 最近变更时间（UTC）
     */
    private LocalDateTime updatedAt;
}
