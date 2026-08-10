package com.stonewu.fusion.controller.system;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.system.vo.RuntimeVersionInfoRespVO;
import com.stonewu.fusion.controller.system.vo.VersionInfoRespVO;
import com.stonewu.fusion.service.system.VersionCheckService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 系统版本 Controller
 */
@Tag(name = "系统版本")
@RestController
@RequestMapping("/api/system/version")
@RequiredArgsConstructor
public class VersionInfoController {

    private final VersionCheckService versionCheckService;

    @Operation(summary = "获取当前版本与最新发布版本")
    @GetMapping
    public CommonResult<VersionInfoRespVO> getVersionInfo(
            @RequestParam(value = "force", required = false, defaultValue = "false") boolean forceRefresh
    ) {
        return CommonResult.success(versionCheckService.getVersionInfo(forceRefresh));
    }

    @Operation(summary = "获取当前运行版本")
    @GetMapping("/runtime")
    public CommonResult<RuntimeVersionInfoRespVO> getRuntimeVersionInfo() {
        return CommonResult.success(versionCheckService.getRuntimeVersionInfo());
    }
}