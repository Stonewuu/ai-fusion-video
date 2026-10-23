package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_generation_attempt")
public class GenerationAttemptEntity {
    /**
     * 提交尝试标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 远端执行记录
     */
    private String jobId;

    /**
     * 经明确决策创建的尝试序号
     */
    private Integer attemptNo;

    /**
     * 可查询协议的预备请求检查点
     */
    private String requestJson;

    /**
     * 不可变供应商受理回执
     */
    private String providerTaskId;

    /**
     * 同步供应商的不可变响应结果
     */
    private String resultJson;

    /**
     * 未知受理的核实结论
     */
    private String resolution;

    /**
     * 核实依据与明确重试说明
     */
    private String resolutionNote;

    /**
     * 作出核实决定的用户
     */
    private Long resolvedBy;

    /**
     * 调用前提交意图时间（UTC）
     */
    private LocalDateTime createdAt;

    /**
     * 准备完成后、付费调用之前提交的意图时间（UTC）
     */
    private LocalDateTime submittedAt;

    /**
     * 回执入库时间（UTC）
     */
    private LocalDateTime acknowledgedAt;

    /**
     * 核实决定时间（UTC）
     */
    private LocalDateTime resolvedAt;
}
