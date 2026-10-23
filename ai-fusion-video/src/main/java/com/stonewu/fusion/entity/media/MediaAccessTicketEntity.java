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
@TableName("afv_media_access_ticket")
public class MediaAccessTicketEntity {
    /**
     * 随机访问凭据的摘要，不保存原文
     */
    @TableId(type = IdType.INPUT)
    private String tokenHash;

    /**
     * 唯一允许访问的媒体标识
     */
    private String resourceId;

    /**
     * 签发时固定的资源版本
     */
    private Integer resourceVersion;

    /**
     * 访问凭据所属用户
     */
    private Long actorUserId;

    /**
     * 签发时固定的团队作用域
     */
    private Long actorTeamId;

    /**
     * 访问用途
     */
    private String purpose;

    /**
     * 是否以附件方式下载
     */
    private Boolean download;

    /**
     * 凭据失效时间
     */
    private LocalDateTime expiresAt;

    /**
     * 凭据是否撤销
     */
    private Boolean revoked;

    /**
     * 签发时间
     */
    private LocalDateTime createdAt;

    /**
     * 绑定的生成执行标识
     */
    private String generationJobId;

    /**
     * 绑定的具体提交尝试
     */
    private String generationAttemptId;

    /**
     * 绑定的智能体运行标识
     */
    private String agentRunId;

    /**
     * 签发时的智能体工作者
     */
    private String agentOwnerInstanceId;

    /**
     * 签发时的智能体领取代数
     */
    private Long agentOwnerEpoch;

    /**
     * 绑定的画布文本任务
     */
    private String canvasTextJobId;
}
