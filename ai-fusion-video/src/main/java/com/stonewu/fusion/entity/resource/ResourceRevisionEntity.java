package com.stonewu.fusion.entity.resource;

import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_resource_revision")
public class ResourceRevisionEntity {
    /**
     * 所属已提交业务命令
     */
    private String commandId;

    /**
     * 业务资源类型
     */
    private String resourceType;

    /**
     * 业务对象编号
     */
    private Long resourceId;

    /**
     * 所属项目编号
     */
    private Long projectId;

    /**
     * 本次命令提交后的对象版本
     */
    private Integer resourceVersion;

    /**
     * 修改前的完整业务内容，新增对象为空
     */
    private String beforeJson;

    /**
     * 修改后的完整业务内容，包含删除状态
     */
    private String afterJson;

    /**
     * 本次实际修改的业务字段
     */
    private String changedFields;

    /**
     * 版本记录时间
     */
    private LocalDateTime createTime;
}
