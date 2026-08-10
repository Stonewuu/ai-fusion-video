package com.stonewu.fusion.entity.storage;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import lombok.*;

/**
 * 存储配置实体
 * <p>
 * 对应数据库表：afv_storage_config
 * 管理文件存储后端配置，支持本地存储和 OSS 等多种类型。
 */
@TableName(value = "afv_storage_config", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StorageConfig extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 配置名称 */
    private String name;

    /** 存储类型：local / s3 */
    private String type;

    /** S3 兼容厂商：generic_s3 / aliyun_oss / tencent_cos / qiniu_kodo / ctyun_zos / minio */
    private String provider;

    /** OSS 端点地址 */
    private String endpoint;

    /** OSS 存储桶名称 */
    private String bucketName;

    /** OSS Access Key */
    private String accessKey;

    /** OSS Secret Key */
    private String secretKey;

    /** 区域 */
    private String region;

    /** 存储根路径（本地为磁盘路径，OSS 为 key 前缀） */
    private String basePath;

    /** 自定义域名（CDN 域名等） */
    private String customDomain;

    /** 厂商扩展配置 JSON */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String options;

    /** 是否为默认存储配置 */
    @Builder.Default
    private Boolean isDefault = false;

    /** 状态：0-禁用 1-启用 */
    @Builder.Default
    private Integer status = 1;

    /** 备注说明 */
    private String remark;
}
