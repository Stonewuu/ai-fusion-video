package com.stonewu.fusion.controller.system;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.*;

@RestController
@RequestMapping("/api/system/execution-recovery")
@PreAuthorize("hasRole('ADMIN')")
@RequiredArgsConstructor
public class ExecutionRecoveryController {
    private final ExecutionRecoveryService recovery;

    @GetMapping
    public CommonResult<Page> list(@RequestParam(defaultValue = "0") int page) {
        return CommonResult.success(recovery.list(SecurityUtils.requireCurrentUserId(), page));
    }

    @GetMapping("/history")
    public CommonResult<List<History>> history(@RequestParam(defaultValue = "0") int page) {
        return CommonResult.success(recovery.history(SecurityUtils.requireCurrentUserId(), page));
    }

    @GetMapping("/{kind}/{id}")
    public CommonResult<View> get(@PathVariable Kind kind, @PathVariable String id) {
        return CommonResult.success(recovery.get(kind, id, SecurityUtils.requireCurrentUserId()));
    }

    @PostMapping("/{kind}/{id}/confirm-finished")
    public CommonResult<Receipt> confirm(@PathVariable Kind kind, @PathVariable String id,
                                         @RequestBody @Valid Decision decision) {
        return CommonResult.success(recovery.confirm(kind, id, decision, SecurityUtils.requireCurrentUserId()));
    }
}
