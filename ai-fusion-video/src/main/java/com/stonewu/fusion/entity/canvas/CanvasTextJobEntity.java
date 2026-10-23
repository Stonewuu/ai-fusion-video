package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_canvas_text_job")
public class CanvasTextJobEntity {
    /**
     * 文本任务标识，也是唯一提交标识
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
     * 所属项目，个人画布可为空
     */
    private Long projectId;

    /**
     * 冻结的文本模型标识
     */
    private Long modelId;

    /**
     * 计划确认时的模型并发上限
     */
    private Integer modelLimit;

    /**
     * 冻结的模型、输入媒体版本和提示词
     */
    private String inputJson;

    /**
     * 持久化文本执行状态
     */
    private String status;

    /**
     * 当前处理阶段
     */
    private String phase;

    /**
     * 付费请求前提交的不可变意图时间
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
     * 已接收的原任务响应，保留迟到响应供核实恢复
     */
    private String resultJson;

    /**
     * 不可变响应摘要
     */
    private String resultHash;

    /**
     * 完整响应接收时间
     */
    private LocalDateTime receivedAt;

    /**
     * 用户是否明确采用已返回的文本
     */
    private Boolean resultAccepted;

    /**
     * 取消意图，不代表远端已经停止
     */
    private Boolean cancelRequested;

    /**
     * 原请求是否仍占用模型额度
     */
    private Boolean permitHeld;

    /**
     * 当前工作者实例
     */
    private String ownerInstanceId;

    /**
     * 工作者领取代数
     */
    private Long ownerEpoch;

    /**
     * 执行租约截止时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 任务状态版本
     */
    private Long revision;

    /**
     * 失败或待核实原因
     */
    private String errorMsg;

    /**
     * 下次领取时间
     */
    private LocalDateTime nextRunAt;

    /**
     * 本次执行截止时间
     */
    private LocalDateTime deadlineAt;

    /**
     * 任务创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 最近状态变更时间
     */
    private LocalDateTime updatedAt;
}
