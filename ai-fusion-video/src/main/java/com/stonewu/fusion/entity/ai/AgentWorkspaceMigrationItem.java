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

@TableName("afv_agent_workspace_migration_item")
@Data
@EqualsAndHashCode(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentWorkspaceMigrationItem extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long migrationId;
    private Long entryId;
    private String sourceBackendType;
    private Long sourceStorageConfigId;
    private String sourceLocalPath;
    private String sourceContentRef;
    private String sourcePayload;
    private String targetBackendType;
    private Long targetStorageConfigId;
    private String targetLocalPath;
    private String targetContentRef;
    private String targetPayload;
    private String contentSha256;
    private Long contentSize;
    private String status;
}
