package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;

@Data
@TableName("afv_media_blob_cleanup")
public class MediaBlobCleanupEntity {
    /**
     * 待清理的独立密文对象标识
     */
    @TableId(type = IdType.INPUT)
    private String blobId;

    /**
     * 等待、保留引用阻止、已领取或已清理
     */
    private String status;

    /**
     * 最早允许物理清理的时间
     */
    private LocalDateTime eligibleAt;

    /**
     * 下次检查或重试时间
     */
    private LocalDateTime nextAttemptAt;

    /**
     * 当前清理工作者标识
     */
    private String ownerId;

    /**
     * 清理领取代数，隔离失效工作者
     */
    private Long epoch;

    /**
     * 领取租约截止时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 实际领取次数
     */
    private Integer attempts;

    /**
     * 不含密钥和存储路径的错误说明
     */
    private String lastError;

    /**
     * 物理文件确认删除的时间
     */
    private LocalDateTime purgedAt;

    /**
     * 首次登记时间
     */
    private LocalDateTime createdAt;

    /**
     * 最近状态变更时间
     */
    private LocalDateTime updatedAt;
}
