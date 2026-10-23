package com.stonewu.fusion.controller.generation.vo;

import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class GenerationScopeReqVO {
    @Valid
    @NotNull(message = "请先确认当前生成账号与团队")
    private GenerationClientScope scope;
}
