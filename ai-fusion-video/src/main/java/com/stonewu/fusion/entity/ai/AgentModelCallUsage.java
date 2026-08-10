package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@TableName("afv_agent_model_call_usage")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentModelCallUsage {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String runId;
    private String modelCallId;
    private String provider;
    private String modelCode;
    private String status;
    private Long inputTokens;
    private Long outputTokens;
    private Long reasoningTokens;
    private Long cacheTokens;
    private String usageJson;
    private String settlementStatus;

    @Builder.Default
    private Integer settlementAttempts = 0;

    private LocalDateTime nextSettlementAttemptAt;
    private String settlementClaimOwner;
    private LocalDateTime settlementClaimUntil;
    private String downstreamSettlementId;
    private String lastSettlementError;
    private LocalDateTime startedAt;
    private LocalDateTime finishedAt;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;

    public NormalizedModelUsage normalizedUsage() {
        return new NormalizedModelUsage(
                inputTokens,
                outputTokens,
                reasoningTokens,
                cacheTokens,
                usageJson);
    }
}
