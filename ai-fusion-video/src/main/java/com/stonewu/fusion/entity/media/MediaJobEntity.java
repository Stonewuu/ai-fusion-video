package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_media_job")
public class MediaJobEntity {
    /**
     * 媒体任务标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 任务类型
     */
    private String kind;

    /**
     * 项目标识，独立任务可为空
     */
    private Long projectId;

    /**
     * 需要正式写回的业务对象标识
     */
    private Long targetId;

    /**
     * 等待写回的业务版本
     */
    private Integer targetVersion;

    /**
     * 发起用户
     */
    private Long actorUserId;

    /**
     * 客户端幂等请求标识
     */
    private String requestId;

    /**
     * 请求与身份摘要
     */
    private String requestHash;

    /**
     * 经认证的执行上下文
     */
    private String contextJson;

    /**
     * 固定的输入来源版本与参数
     */
    private String inputJson;

    /**
     * 已持久化的阶段结果
     */
    private String checkpointJson;

    /**
     * 任务状态
     */
    private String status;

    /**
     * 当前处理阶段
     */
    private String phase;

    /**
     * 终止或待处理原因
     */
    private String errorMsg;

    /**
     * 当前执行实例
     */
    private String ownerInstanceId;

    /**
     * 执行代数
     */
    private Long ownerEpoch;

    /**
     * 执行租约截止时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 下次执行时间
     */
    private LocalDateTime nextRunAt;

    /**
     * 任务总时限
     */
    private LocalDateTime deadlineAt;

    /**
     * 是否请求取消
     */
    private Boolean cancelRequested;

    /**
     * 状态版本
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
