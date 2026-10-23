package com.stonewu.fusion.entity.execution;

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
@TableName("afv_execution_recovery_decision")
public class ExecutionRecoveryDecisionEntity {
    /**
     * 管理员核实决定标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 实际操作的管理员编号
     */
    private Long adminUserId;

    /**
     * 管理员请求幂等标识
     */
    private String requestId;

    /**
     * 具体核实决定的不可变摘要
     */
    private String requestHash;

    /**
     * 被核实的执行类型
     */
    private String executionKind;

    /**
     * 被核实的持久化任务标识
     */
    private String jobId;

    /**
     * 管理员看到并确认的任务版本
     */
    private Long expectedRevision;

    /**
     * 原请求发起用户，禁止改写为管理员
     */
    private Long actorUserId;

    /**
     * 原请求发起团队作用域
     */
    private Long actorTeamId;

    /**
     * 原任务所属项目
     */
    private Long projectId;

    /**
     * 原任务所属画布
     */
    private String canvasId;

    /**
     * 核实前的脱敏状态与远端标识，不含提示词、结果内容或密钥
     */
    private String factsJson;

    /**
     * 管理员核实远端已结束的依据
     */
    private String note;

    /**
     * 同一事务提交的任务收尾回执
     */
    private String receiptJson;

    /**
     * 核实完成时间（UTC）
     */
    private LocalDateTime createdAt;
}
