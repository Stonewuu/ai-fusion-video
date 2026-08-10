package com.stonewu.fusion.controller.storage.vo;

import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * 存储配置响应 VO
 */
@Data
public class StorageConfigRespVO {

    private Long id;
    private String name;
    private String type;
    private String provider;
    private String endpoint;
    private String bucketName;
    private String accessKey;
    private String secretKey;
    private String region;
    private String basePath;
    private String customDomain;
    private Map<String, Object> options;
    private Boolean isDefault;
    private Integer status;
    private String remark;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
