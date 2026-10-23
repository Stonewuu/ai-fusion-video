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
@TableName("afv_canvas_run")
public class CanvasRunEntity {
    /**
     * 画布运行标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属画布
     */
    private String canvasId;

    /**
     * 确认的计划
     */
    private String planId;

    /**
     * 发起用户
     */
    private Long actorUserId;

    /**
     * 发起团队作用域
     */
    private Long actorTeamId;

    /**
     * 请求身份作用域摘要
     */
    private String scopeKey;

    /**
     * 启动请求幂等标识
     */
    private String requestId;

    /**
     * 启动请求摘要
     */
    private String requestHash;

    /**
     * 经过认证的调用来源
     */
    private String contextJson;

    /**
     * 本次运行的完整冻结计划
     */
    private String snapshotJson;

    /**
     * 运行状态
     */
    private String status;

    /**
     * 是否停止后续执行
     */
    private Boolean cancelRequested;

    /**
     * 运行状态版本
     */
    private Long revision;

    /**
     * 运行错误或停止原因
     */
    private String errorMsg;

    /**
     * 运行执行截止时间
     */
    private LocalDateTime deadlineAt;

    /**
     * 受理时间
     */
    private LocalDateTime createdAt;

    /**
     * 最近状态变更时间
     */
    private LocalDateTime updatedAt;
}
