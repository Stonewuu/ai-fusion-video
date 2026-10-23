package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.FieldStrategy;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
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
@TableName("afv_generation_submission")
public class GenerationSubmissionEntity {
    /**
     * 受理记录标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 调用方幂等标识
     */
    private String requestId;

    /**
     * 用户团队与入口作用域摘要
     */
    private String scopeKey;

    /**
     * 规范化请求摘要
     */
    private String requestHash;

    /**
     * 生成类别
     */
    private String kind;

    /**
     * 业务任务数据库标识
     */
    private Long taskId;

    /**
     * 业务任务公开标识
     */
    private String publicTaskId;

    /**
     * 发起用户
     */
    private Long actorUserId;

    /**
     * 发起团队上下文
     */
    private Long actorTeamId;

    /**
     * 所属项目
     */
    private Long projectId;

    /**
     * 画布节点运行标识
     */
    private String canvasNodeRunId;

    /**
     * 经校验的内部执行身份
     */
    private String contextJson;

    /**
     * 模型与能力的不可变快照
     */
    private String modelSnapshotJson;

    /**
     * 供应商配置标识
     */
    private Long apiConfigId;

    /**
     * 供应商配置一致性摘要
     */
    private String apiConfigHash;

    /**
     * 受理及聚合执行状态
     */
    private String state;

    /**
     * 是否请求停止后续执行
     */
    private Boolean cancelRequested;

    /**
     * 状态版本
     */
    private Long revision;

    /**
     * 受理时间（UTC）
     */
    private LocalDateTime createdAt;

    /**
     * 最近状态变化时间（UTC）
     */
    private LocalDateTime updatedAt;

    /**
     * 智能体运行与工具调用的稳定身份摘要
     */
    @TableField(insertStrategy = FieldStrategy.NEVER, updateStrategy = FieldStrategy.NEVER)
    private String agentInvocationKey;

    /**
     * 画布执行与智能体工具调用的唯一受理身份
     */
    @TableField(insertStrategy = FieldStrategy.NEVER, updateStrategy = FieldStrategy.NEVER)
    private String canvasInvocationKey;
}
