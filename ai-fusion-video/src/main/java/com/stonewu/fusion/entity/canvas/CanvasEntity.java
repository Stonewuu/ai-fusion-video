package com.stonewu.fusion.entity.canvas;

import com.baomidou.mybatisplus.annotation.TableField;

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
@TableName("afv_canvas")
public class CanvasEntity {
    /**
     * 画布唯一标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 画布名称
     */
    private String title;

    /**
     * 关联项目编号
     */
    private Long projectId;

    /**
     * 所有者类型：1个人，2团队
     */
    private Integer ownerType;

    /**
     * 所有者编号
     */
    private Long ownerId;

    /**
     * 创建用户编号
     */
    private Long createdBy;

    /**
     * 画布协议版本
     */
    private Integer schemaVersion;

    /**
     * 已提交文档版本
     */
    private Long revision;

    /**
     * 下一条提交事件序号
     */
    private Long nextEventSequence;

    /**
     * 是否已归档
     */
    private Boolean archived;

    /**
     * 创建时间
     */
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    private LocalDateTime updateTime;
    @TableField(exist = false)
    private String projectName;
}
