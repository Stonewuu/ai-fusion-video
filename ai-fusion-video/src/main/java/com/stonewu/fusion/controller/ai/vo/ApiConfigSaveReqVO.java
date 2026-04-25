package com.stonewu.fusion.controller.ai.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Schema(description = "API配置保存请求")
@Data
public class ApiConfigSaveReqVO {
    private Long id;
    @NotBlank(message = "配置名称不能为空")
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
}
