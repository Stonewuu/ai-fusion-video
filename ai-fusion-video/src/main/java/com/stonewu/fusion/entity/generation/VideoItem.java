package com.stonewu.fusion.entity.generation;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 生视频条目实体
 * <p>
 * 对应数据库表：afv_video_item
 * 记录每次视频生成任务产出的单个视频结果。一个生视频任务（VideoTask）可对应多条视频记录。
 */
@TableName("afv_video_item")
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VideoItem extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属生视频任务ID */
    private Long taskId;

    /** 平台侧任务ID，用于回调和状态查询 */
    private String platformTaskId;

    /** 生成的视频URL */
    private String videoUrl;

    /** 视频封面图URL */
    private String coverUrl;

    /** 视频时长（秒） */
    private Integer duration;

    /** 文件大小（字节） */
    private Long fileSize;

    /** 状态：0-生成中 1-成功 2-失败 */
    @Builder.Default
    private Integer status = 0;

    /** 失败时的错误信息 */
    private String errorMsg;

    /** 视频首帧图片URL */
    private String firstFrameUrl;

    /** 视频尾帧图片URL */
    private String lastFrameUrl;
}
