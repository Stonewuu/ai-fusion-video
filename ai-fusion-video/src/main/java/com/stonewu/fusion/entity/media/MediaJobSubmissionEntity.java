package com.stonewu.fusion.entity.media;

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
@TableName("afv_media_job_submission")
public class MediaJobSubmissionEntity {
    /**
     * 媒体任务标识
     */
    private String jobId;

    /**
     * 付费提交步骤
     */
    private String step;

    /**
     * 唯一提交尝试
     */
    private String attemptId;

    /**
     * 供应商确认的任务标识
     */
    private String providerTaskId;

    /**
     * 提交意图登记时间
     */
    private LocalDateTime createdAt;

    /**
     * 供应商回执登记时间
     */
    private LocalDateTime acknowledgedAt;
}
