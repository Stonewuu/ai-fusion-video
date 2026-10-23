package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.ai.configuration.AgentSettingsOverride;
import com.stonewu.fusion.service.ai.configuration.AgentUserConfigurationService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/ai/agent-config/definitions")
@RequiredArgsConstructor
public class AgentDefinitionConfigurationController {
    public record Save(@Min(0) long expectedRevision, @NotNull @Valid AgentSettingsOverride overrides) {
    }

    private final AgentUserConfigurationService configurations;

    @GetMapping
    public CommonResult<List<AgentUserConfigurationService.Summary>> list() {
        return CommonResult.success(configurations.list(SecurityUtils.requireCurrentUserId()));
    }

    @GetMapping("/{type}")
    public CommonResult<AgentUserConfigurationService.Detail> get(@PathVariable String type) {
        return CommonResult.success(configurations.detail(SecurityUtils.requireCurrentUserId(), type));
    }

    @PutMapping("/{type}")
    public CommonResult<AgentUserConfigurationService.Detail> save(@PathVariable String type,
                                                                   @Valid @RequestBody Save request) {
        long userId = SecurityUtils.requireCurrentUserId();
        configurations.save(userId, type, request.expectedRevision(), request.overrides());
        return CommonResult.success(configurations.detail(userId, type));
    }
}
