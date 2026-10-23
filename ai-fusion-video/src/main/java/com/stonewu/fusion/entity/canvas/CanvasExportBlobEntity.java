package com.stonewu.fusion.entity.canvas;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.ToString;

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
@TableName("afv_canvas_export_blob")
public class CanvasExportBlobEntity {
    /**
     * 加密交付文件编号
     */
    @TableId(type = IdType.INPUT)
    private String id;

    /**
     * 所属交付任务编号
     */
    private String exportId;

    /**
     * 存储配置编号
     */
    private Long storageConfigId;

    /**
     * 存储后端类型
     */
    private String storageBackend;

    /**
     * 加密存储对象键
     */
    private String objectKey;

    /**
     * 仅服务端数据库保存的流式加密密钥集合
     */
    @JsonIgnore
    @ToString.Exclude
    private byte[] encryptionKeyset;

    /**
     * 文件登记、完成、孤立或清理状态
     */
    private String state;

    /**
     * 创建文件的工作者隔离代次
     */
    private Long ownerEpoch;

    /**
     * 加密存储字节数
     */
    private Long storedBytes;

    /**
     * 登记时间
     */
    private LocalDateTime createTime;

    /**
     * 状态更新时间
     */
    private LocalDateTime updateTime;
}
