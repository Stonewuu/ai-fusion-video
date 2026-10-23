package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_canvas_mcp_job")
public class CanvasMcpJobEntity {
    /**
     * 唯一远端提交标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属画布节点运行
     */
    private String nodeRunId;

    /**
     * 所属画布运行
     */
    private String runId;

    /**
     * 持久化的原调用用户
     */
    private Long userId;

    /**
     * 冻结的工具契约标识
     */
    private String contractId;

    /**
     * 冻结的契约、参数和输入来源，不含连接凭据
     */
    private String inputJson;

    /**
     * 持久化调用状态
     */
    private String status;

    /**
     * 远端调用前提交的不可变意图时间
     */
    private LocalDateTime requestStartedAt;

    /**
     * 唯一提交尝试的原工作者
     */
    private String requestOwnerInstanceId;

    /**
     * 唯一提交尝试的原领取代数
     */
    private Long requestOwnerEpoch;

    /**
     * 不可变的原响应与契约校验结果
     */
    private String resultJson;

    /**
     * 已登记响应摘要
     */
    private String resultHash;

    /**
     * 完整响应接收时间
     */
    private LocalDateTime receivedAt;

    /**
     * 取消意图，不代表远端副作用已撤回
     */
    private Boolean cancelRequested;

    /**
     * 原调用是否占用该契约的串行执行名额
     */
    private Boolean permitHeld;

    /**
     * 当前工作者实例
     */
    private String ownerInstanceId;

    /**
     * 当前领取代数
     */
    private Long ownerEpoch;

    /**
     * 执行租约截止时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 状态与核实版本
     */
    private Long revision;

    /**
     * 失败或待核实原因
     */
    private String errorMsg;

    /**
     * 本次任务截止时间
     */
    private LocalDateTime deadlineAt;

    /**
     * 创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 最近状态变更时间
     */
    private LocalDateTime updatedAt;
}
