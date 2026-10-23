package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

import lombok.Data;
import lombok.ToString;
import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_media_blob")
public class MediaBlobEntity {
    /**
     * 每轮接收独立的密文对象标识
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 对应媒体资源标识
     */
    private String resourceId;

    /**
     * 所属接收代数
     */
    private Long ingestEpoch;

    /**
     * 实际使用的存储配置编号
     */
    private Long storageConfigId;

    /**
     * 已固定的存储后端
     */
    private String backend;

    /**
     * 密文对象相对位置，不可作为明文地址
     */
    private String objectKey;

    /**
     * 仅服务端可读的流式密钥集，禁止接口返回、缓存或日志输出
     */
    @JsonIgnore
    @ToString.Exclude
    private byte[] keysetBytes;

    /**
     * 写入完成后的密文字节数
     */
    private Long storedBytes;

    /**
     * 对象接收与保留状态
     */
    private String state;

    /**
     * 预留时间
     */
    private LocalDateTime createdAt;

    /**
     * 确认接收失败或被新接收代数替代的时间
     */
    private LocalDateTime orphanedAt;
}
