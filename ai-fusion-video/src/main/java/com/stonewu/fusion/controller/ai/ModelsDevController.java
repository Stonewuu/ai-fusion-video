package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.ai.catalog.ModelsDevCatalogService;
import com.stonewu.fusion.service.ai.catalog.ModelsDevMatcher;
import com.stonewu.fusion.service.ai.catalog.ModelsDevMatcher.SearchResult;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/ai/model/models-dev")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
@Validated
public class ModelsDevController {
    private final ModelsDevCatalogService catalog;

    @GetMapping("/search")
    public CommonResult<SearchResult> search(@RequestParam(defaultValue = "") @Size(max = 120) String query,
                                           @RequestParam @Min(1) @Max(4) int modelType,
                                           @RequestParam(required = false) Long apiConfigId,
                                           @RequestParam(required = false) @Size(max = 120) String providerId) {
        return CommonResult.success(ModelsDevMatcher.search(catalog.load(apiConfigId), query, modelType, providerId));
    }

    @PostMapping("/refresh")
    public CommonResult<Boolean> refresh() {
        catalog.refresh();
        return CommonResult.success(true);
    }
}
