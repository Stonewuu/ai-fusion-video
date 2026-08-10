package com.stonewu.fusion.controller.asset.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 更新子资产请求 VO
 */
@Schema(description = "更新子资产请求")
@Data
public class AssetItemUpdateReqVO {

    @NotNull(message = "子资产ID不能为空")
    private Long id;

    private String itemType;
    private String name;
    private String imageUrl;
    private String thumbnailUrl;
    private String properties;
    private Integer sortOrder;
    private String aiPrompt;
}
