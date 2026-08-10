package com.stonewu.fusion.controller.asset.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 更新资产请求 VO
 */
@Schema(description = "更新资产请求")
@Data
public class AssetUpdateReqVO {

    @NotNull(message = "资产ID不能为空")
    private Long id;

    private String type;
    private String name;
    private String description;
    private String coverUrl;
    private String properties;
    private String tags;
    private String aiPrompt;
}
