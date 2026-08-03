package com.stonewu.fusion.controller.script.vo;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/** 替换剧本原文并重置结构化内容请求。 */
@Data
public class ScriptSourceReplaceReqVO {

    @NotBlank(message = "剧本原文不能为空")
    private String rawContent;
}
