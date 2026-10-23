package com.stonewu.fusion.entity.ai;

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
@TableName("afv_comfyui_test_run")
public class ComfyUiTestRunEntity {
    /**
     * 共享媒体任务标识
     */
    @TableId(type = IdType.INPUT)
    private String jobId;

    /**
     * 工作流标识
     */
    private Long workflowId;

    /**
     * 试运行的工作流版本
     */
    private Long versionId;

    /**
     * 受理时的工作流内容摘要
     */
    private String workflowHash;

    /**
     * 目标供应商配置
     */
    private Long apiConfigId;

    /**
     * 供应商连接一致性摘要
     */
    private String apiConfigHash;

    /**
     * 受理时的团队作用域
     */
    private Long actorTeamId;

    /**
     * 提交前固定的远端请求标识
     */
    private String promptId;

    /**
     * 已准备的远端请求，禁止直接暴露给客户端
     */
    private String requestJson;

    /**
     * 实际提交意图时间（UTC）
     */
    private LocalDateTime submittedAt;

    /**
     * 不可变受理回执时间（UTC）
     */
    private LocalDateTime acknowledgedAt;

    /**
     * 已确认的远端结果检查点
     */
    private String resultJson;

    /**
     * 远端任务是否仍占用试运行额度
     */
    private Boolean permitHeld;

    /**
     * 是否已经发送远端取消请求
     */
    private Boolean cancelSent;

    /**
     * 单次查询次数
     */
    private Integer pollAttempts;

    /**
     * 连续查询失败次数
     */
    private Integer consecutiveErrors;
}
