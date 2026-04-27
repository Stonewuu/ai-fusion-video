package com.stonewu.fusion.controller.ai.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import java.time.LocalDateTime;

@Schema(description = "API配置响应")
@Data
public class ApiConfigRespVO {
    private Long id;
    private String name;
    private String platform;
    private String apiUrl;
    private Boolean autoAppendV1Path;
    private String proxyType;
    private String proxyHost;
    private Integer proxyPort;
    private String proxyUsername;
    private String proxyPassword;
    private String apiKey;
    private String appId;
    private String appSecret;
    private Long modelId;
    private Integer status;
    private String remark;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
