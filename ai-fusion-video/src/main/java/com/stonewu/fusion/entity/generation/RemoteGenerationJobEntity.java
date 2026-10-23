package com.stonewu.fusion.entity.generation;

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
@TableName("afv_remote_generation_job")
public class RemoteGenerationJobEntity {
    /**
     * 远端执行记录标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属受理记录
     */
    private String submissionId;

    /**
     * 对应的结果条目标识
     */
    private Long itemId;

    /**
     * 请求中的稳定结果顺序
     */
    private Integer itemIndex;

    /**
     * 占用额度的模型
     */
    private Long modelId;

    /**
     * 占用额度的用户
     */
    private Long actorUserId;

    /**
     * 占用额度的项目
     */
    private Long projectId;

    /**
     * 远端执行状态
     */
    private String state;

    /**
     * 当前持久化执行阶段
     */
    private String phase;

    /**
     * 供应商受理标识
     */
    private String providerTaskId;

    /**
     * 当前显式提交尝试标识
     */
    private String activeAttemptId;

    /**
     * 固定生成参数与稳定媒体引用
     */
    private String inputJson;

    /**
     * 供应商结果接收检查点
     */
    private String resultJson;

    /**
     * 是否请求远端取消
     */
    private Boolean cancelRequested;

    /**
     * 是否已发送远端取消请求
     */
    private Boolean cancelSent;

    /**
     * 是否仍占用远端并发额度
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
     * 租约到期时间（UTC）
     */
    private LocalDateTime leaseUntil;

    /**
     * 下次执行时间（UTC）
     */
    private LocalDateTime nextPollAt;

    /**
     * 自动等待截止时间（UTC）
     */
    private LocalDateTime deadlineAt;

    /**
     * 单次查询累计次数
     */
    private Integer pollAttempts;

    /**
     * 连续传输错误次数
     */
    private Integer consecutiveErrors;

    /**
     * 稳定错误码
     */
    private String errorCode;

    /**
     * 用户可读执行原因
     */
    private String errorMessage;

    /**
     * 状态版本
     */
    private Long revision;

    /**
     * 创建时间（UTC）
     */
    private LocalDateTime createdAt;

    /**
     * 最近变更时间（UTC）
     */
    private LocalDateTime updatedAt;
}
