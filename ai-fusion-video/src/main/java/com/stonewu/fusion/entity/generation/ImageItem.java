package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.*;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 生图条目实体
 * <p>
 * 对应数据库表：afv_image_item
 * 记录每次图片生成任务产出的单张图片结果。一个生图任务（ImageTask）可对应多条图片记录。
 */
@TableName("afv_image_item")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ImageItem extends BaseEntity {

    /**
     * 主键ID，自增
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 生成结果内容版本，写入必须匹配读取版本。
     */
    @Builder.Default
    @TableField(updateStrategy = FieldStrategy.NEVER)
    private Integer version = 0;

    /**
     * 所属生图任务ID
     */
    @TableField(updateStrategy = FieldStrategy.NEVER)
    private Long taskId;

    /**
     * 平台侧任务ID，用于回调和状态查询
     */
    private String platformTaskId;

    /**
     * 生成的图片URL
     */
    private String imageUrl;

    /**
     * 缩略图URL
     */
    private String thumbnailUrl;

    /**
     * 图片宽度（像素）
     */
    private Integer width;

    /**
     * 图片高度（像素）
     */
    private Integer height;

    /**
     * 文件大小（字节）
     */
    private Long fileSize;

    /**
     * 状态：0-生成中 1-成功 2-失败
     */
    @Builder.Default
    private Integer status = 0;

    /**
     * 失败时的错误信息
     */
    private String errorMsg;
}
