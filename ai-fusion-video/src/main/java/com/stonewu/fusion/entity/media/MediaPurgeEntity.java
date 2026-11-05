package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("afv_media_purge")
public class MediaPurgeEntity {
    @TableId(type = IdType.INPUT)
    private String blobId;
    private String mediaId;
    private String state;
    private Integer attempts;
    private LocalDateTime nextAttemptAt;
    private LocalDateTime updatedAt;
}
