package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_media_preview_job")
public class MediaPreviewJobEntity {
    /**
     * 预览生成任务标识
     */
    @TableId(type = IdType.INPUT)
    private String jobId;

    /**
     * 原始媒体资源标识
     */
    private String sourceResourceId;

    /**
     * 固定的原始媒体版本
     */
    private Integer sourceVersion;

    /**
     * 预览规格及算法版本
     */
    private String profile;

    /**
     * 视频抽样位置，图片固定为零
     */
    private Long sampleAtMs;

    /**
     * 显式重试的上一任务标识
     */
    private String previousJobId;

    /**
     * 受原始媒体权限约束的预览资源
     */
    private String outputResourceId;
}
