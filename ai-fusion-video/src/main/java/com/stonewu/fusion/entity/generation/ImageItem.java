package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
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

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属生图任务ID */
    private Long taskId;

    /** 平台侧任务ID，用于回调和状态查询 */
    private String platformTaskId;

    /** 生成的图片URL */
    private String imageUrl;

    /** 缩略图URL */
    private String thumbnailUrl;

    /** 图片宽度（像素） */
    private Integer width;

    /** 图片高度（像素） */
    private Integer height;

    /** 文件大小（字节） */
    private Long fileSize;

    /** 状态：0-生成中 1-成功 2-失败 */
    @Builder.Default
    private Integer status = 0;

    /** 失败时的错误信息 */
    private String errorMsg;
}
