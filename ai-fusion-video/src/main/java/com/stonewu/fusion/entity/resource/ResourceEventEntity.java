package com.stonewu.fusion.entity.resource;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import com.baomidou.mybatisplus.annotation.TableField;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_resource_event")
public class ResourceEventEntity {
    /**
     * 资源事件顺序编号
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 资源事件唯一标识
     */
    private String eventId;

    /**
     * 所属业务命令标识
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
     * 提交后的业务版本
     */
    private Integer resourceVersion;

    /**
     * 对象是否已删除
     */
    private Boolean deleted;

    /**
     * 本次变更的字段名称
     */
    private String changedFields;

    /**
     * 是否已投递至相关画布
     */
    private Boolean published;

    /**
     * 事件提交时间
     */
    private LocalDateTime createTime;

    /**
     * 完成画布投递的时间
     */
    private LocalDateTime publishedAt;

    @TableField(exist = false)
    private Long sequenceNo;
}
