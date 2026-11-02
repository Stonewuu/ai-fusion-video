package com.stonewu.fusion.controller.system;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.execution.ExecutionRecordQueries;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Kind;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/execution-records")
@RequiredArgsConstructor
public class ExecutionRecordController {
    private final ExecutionRecordQueries records;

    @GetMapping
    public CommonResult<ExecutionRecordQueries.Page> list(@RequestParam(defaultValue = "0") int page) {
        return CommonResult.success(records.list(SecurityUtils.requireCurrentUserId(), page));
    }

    @GetMapping("/{kind}/{id}")
    public CommonResult<ExecutionRecordQueries.Detail> get(@PathVariable Kind kind, @PathVariable String id) {
        return CommonResult.success(records.get(SecurityUtils.requireCurrentUserId(), kind, id));
    }
}
