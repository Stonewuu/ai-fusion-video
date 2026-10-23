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
@TableName("afv_canvas_comment")
public class CanvasCommentEntity {
    /**
     * 评论唯一标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 画布标识
     */
    private String canvasId;

    /**
     * 锚定节点标识
     */
    private String nodeId;

    /**
     * 被回复评论标识
     */
    private String parentId;

    /**
     * 评论对应的业务内容版本
     */
    private Long resourceVersion;

    /**
     * 评论内容
     */
    private String content;

    /**
     * 评论用户编号
     */
    private Long createdBy;

    /**
     * 是否已解决
     */
    private Boolean resolved;

    /**
     * 评论版本
     */
    private Long version;

    /**
     * 是否已删除评论
     */
    private Boolean deleted;

    /**
     * 创建时间
     */
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    private LocalDateTime updateTime;
    @TableField(exist = false)
    private String authorName;
}
