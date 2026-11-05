package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("afv_media_deletion")
public class MediaDeletionEntity {
    @TableId(type = IdType.INPUT)
    private String mediaId;
    private String mode;
    private String requestId;
    private String requestHash;
    private Long actorUserId;
    private Long actorTeamId;
    private Integer resourceVersion;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
