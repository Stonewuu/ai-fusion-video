package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@TableName("afv_agent_state_cleanup_policy")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentStateCleanupPolicy {

    @TableId(type = IdType.INPUT)
    private Long id;
    private Integer cleanupIntervalDays;
    private Integer retentionDays;
    private LocalDateTime nextCleanupAt;
    @TableField(updateStrategy = FieldStrategy.ALWAYS)
    private String cleanupLeaseOwner;

    @TableField(updateStrategy = FieldStrategy.ALWAYS)
    private LocalDateTime cleanupLeaseUntil;
    private LocalDateTime lastCleanupAt;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}
