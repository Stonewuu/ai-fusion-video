package com.stonewu.fusion.controller.storage.vo;

import lombok.Data;

import java.util.Map;

/**
 * 存储配置保存请求 VO
 */
@Data
public class StorageConfigSaveReqVO {

    /** ID（更新时必传） */
    private Long id;

    /** 配置名称 */
    private String name;

    /** 存储类型：local / s3 */
    private String type;

    /** S3 兼容厂商 */
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

    /** 存储根路径 */
    private String basePath;

    /** 自定义域名 */
    private String customDomain;

    /** 厂商扩展配置 */
    private Map<String, Object> options;

    /** 是否为默认存储配置 */
    private Boolean isDefault;

    /** 状态 */
    private Integer status;

    /** 备注 */
    private String remark;
}
