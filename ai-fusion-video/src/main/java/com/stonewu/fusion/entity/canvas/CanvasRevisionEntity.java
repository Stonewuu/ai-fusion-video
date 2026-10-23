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
@TableName("afv_canvas_revision")
public class CanvasRevisionEntity {
    /**
     * 检查点唯一标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 画布标识
     */
    private String canvasId;

    /**
     * 检查点名称
     */
    private String title;

    /**
     * 对应文档版本
     */
    private Long revision;

    /**
     * 对应已提交事件序号
     */
    private Long sequenceNo;

    /**
     * 图结构与业务版本引用快照
     */
    private String snapshotJson;

    /**
     * 创建用户编号
     */
    private Long createdBy;

    /**
     * 创建时间
     */
    private LocalDateTime createTime;
    @TableField(exist = false)
    private String authorName;
}
