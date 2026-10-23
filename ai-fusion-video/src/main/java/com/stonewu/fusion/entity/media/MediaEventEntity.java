package com.stonewu.fusion.entity.media;

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
@TableName("afv_media_event")
public class MediaEventEntity {
    /**
     * 媒体事件顺序编号
     */
    @TableId(type = IdType.AUTO)
    private Long sequenceNo;

    /**
     * 媒体事件唯一标识
     */
    private String eventId;

    /**
     * 媒体资源标识
     */
    private String resourceId;

    /**
     * 提交后的资源版本
     */
    private Integer resourceVersion;

    /**
     * 媒体事件类型
     */
    private String eventType;

    /**
     * 经过认证的操作用户编号
     */
    private Long actorUserId;

    /**
     * 所属项目编号
     */
    private Long projectId;

    /**
     * 不含密钥、凭据和存储配置的事件正文
     */
    private String payloadJson;

    /**
     * 是否已向相关画布投递
     */
    private Boolean published;

    /**
     * 事件提交时间
     */
    private LocalDateTime createdAt;
}
