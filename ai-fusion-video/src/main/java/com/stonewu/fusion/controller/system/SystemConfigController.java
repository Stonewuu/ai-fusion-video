package com.stonewu.fusion.controller.system;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.entity.system.SystemConfig;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.system.SystemConfigService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 系统配置 Controller
 */
@Tag(name = "系统配置")
@RestController
@RequestMapping("/api/system/config")
@RequiredArgsConstructor
public class SystemConfigController {

    private final SystemConfigService systemConfigService;

    @Operation(summary = "获取所有系统配置")
    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<List<SystemConfig>> list() {
        return CommonResult.success(systemConfigService.getAll());
    }

    @Operation(summary = "获取单个配置值")
    @GetMapping("/{key}")
    @PreAuthorize("hasRole('ADMIN') or #key == 'resource_base_url'")
    public CommonResult<String> get(@PathVariable String key) {
        return CommonResult.success(systemConfigService.getValue(key));
    }

    @Operation(summary = "保存配置（批量）")
    @PutMapping
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> save(@RequestBody Map<String, String> configs) {
        configs.forEach((key, value) -> {
            if (key.startsWith("art_preset_url:") && MediaReference.managed(value)) {
                throw new BusinessException(400, "托管画风引用请使用内置画风上传入口，不能发布任意私人素材");
            }
        });
        configs.forEach(systemConfigService::setValue);
        return CommonResult.success(true);
    }
}
