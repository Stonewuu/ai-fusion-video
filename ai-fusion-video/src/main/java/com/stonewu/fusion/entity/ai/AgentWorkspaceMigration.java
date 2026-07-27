package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@TableName("afv_agent_workspace_migration")
@Data
@EqualsAndHashCode(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentWorkspaceMigration extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;
    private String sourceBackendType;
    private Long sourceStorageConfigId;
    private String sourceLocalPath;
    private String targetBackendType;
    private Long targetStorageConfigId;
    private String targetLocalPath;
    private String status;
    private Long totalCount;
    private Long copiedCount;
    private Long failedCount;
    private String errorMessage;
    private LocalDateTime startedAt;
    private LocalDateTime finishedAt;
}
