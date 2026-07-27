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

@TableName("afv_agent_workspace_entry")
@Data
@EqualsAndHashCode(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentWorkspaceEntry extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;
    private String namespaceHash;
    private String namespaceKey;
    private String itemHash;
    private String itemKey;
    private String backendType;
    private Long storageConfigId;
    private String localPath;
    private String contentRef;
    private String payload;
    private String contentSha256;
    private Long contentSize;
    private Long version;
}
