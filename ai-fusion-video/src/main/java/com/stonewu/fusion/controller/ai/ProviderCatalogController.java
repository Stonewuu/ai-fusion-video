package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.ai.vo.ApiConfigSaveReqVO;
import com.stonewu.fusion.controller.ai.vo.ProviderSetupReqVO;
import com.stonewu.fusion.service.ai.catalog.ProviderDiscoveryResult;
import com.stonewu.fusion.service.ai.catalog.ProviderDiscoveryService;
import com.stonewu.fusion.service.ai.catalog.ProviderSetupService;
import com.stonewu.fusion.service.ai.catalog.ProviderSetupResult;
import jakarta.validation.Valid;
import com.stonewu.fusion.service.ai.catalog.ProviderCatalogService;
import com.stonewu.fusion.service.ai.catalog.ProviderDefinition;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/ai/providers")
@RequiredArgsConstructor
public class ProviderCatalogController {
    private final ProviderCatalogService catalog;
    private final ProviderDiscoveryService discovery;
    private final ProviderSetupService setup;

    @GetMapping
    public CommonResult<List<ProviderDefinition>> list() {
        return CommonResult.success(catalog.list());
    }

    @PostMapping("/preview")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<ProviderDiscoveryResult> preview(@RequestBody ApiConfigSaveReqVO request) {
        return CommonResult.success(discovery.preview(request));
    }

    @PostMapping("/setup")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<ProviderSetupResult> setup(@Valid @RequestBody ProviderSetupReqVO request) {
        return CommonResult.success(setup.apply(request));
    }
}
