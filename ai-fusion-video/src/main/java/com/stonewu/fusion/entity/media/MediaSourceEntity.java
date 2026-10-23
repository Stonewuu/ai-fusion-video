package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_media_source")
public class MediaSourceEntity {
    /**
     * 媒体资源标识
     */
    private String resourceId;

    /**
     * 经过授权的来源业务类型
     */
    private String sourceType;

    /**
     * 来源业务对象标识
     */
    private String sourceId;

    /**
     * 登记时固定的来源版本
     */
    private Integer sourceVersion;

    /**
     * 显式映射的来源字段
     */
    private String sourceField;

    /**
     * 来源登记时间
     */
    private LocalDateTime createdAt;
}
