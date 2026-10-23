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
@TableName("afv_media_resource")
public class MediaResourceEntity {
    /**
     * 不可变媒体资源标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属项目编号，独立素材为空
     */
    private Long projectId;

    /**
     * 所有者类型：一为用户，二为团队
     */
    private Integer ownerType;

    /**
     * 所有者编号
     */
    private Long ownerId;

    /**
     * 经过认证的创建用户编号
     */
    private Long createdBy;

    /**
     * 创建时固定的团队作用域
     */
    private Long actorTeamId;

    /**
     * 创建用户作用域内的幂等键
     */
    private String requestId;

    /**
     * 内容及来源作用域摘要
     */
    private String requestHash;

    /**
     * 媒体类别
     */
    private String mediaKind;

    /**
     * 经内容检测的媒体类型
     */
    private String mimeType;

    /**
     * 安全处理后的原始文件名
     */
    private String originalName;

    /**
     * 明文内容字节数
     */
    private Long byteSize;

    /**
     * 明文内容校验摘要
     */
    private String contentHash;

    /**
     * 画面宽度
     */
    private Integer width;

    /**
     * 画面高度
     */
    private Integer height;

    /**
     * 持续时间毫秒数
     */
    private Long durationMs;

    /**
     * 是否包含音轨
     */
    private Boolean hasAudio;

    /**
     * 登记状态
     */
    private String state;

    /**
     * 资源元数据版本
     */
    private Integer version;

    /**
     * 已完成或正在接收的密文对象标识
     */
    private String blobId;

    /**
     * 接收者隔离代数
     */
    private Long ingestEpoch;

    /**
     * 本轮接收租约到期时间
     */
    private LocalDateTime leaseUntil;

    /**
     * 可向用户显示的接收失败原因
     */
    private String lastError;

    /**
     * 可信操作来源
     */
    private String origin;

    /**
     * 发起操作的智能体运行标识
     */
    private String agentRunId;

    /**
     * 发起操作的工具调用标识
     */
    private String toolCallId;

    /**
     * 发起操作的画布标识
     */
    private String canvasId;

    /**
     * 创建时间
     */
    private LocalDateTime createdAt;

    /**
     * 最近状态变更时间
     */
    private LocalDateTime updatedAt;
}
