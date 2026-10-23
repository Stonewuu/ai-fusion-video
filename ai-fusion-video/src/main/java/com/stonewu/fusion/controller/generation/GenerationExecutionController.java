package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationJobCommands;
import com.stonewu.fusion.service.generation.job.GenerationJobQueries;
import com.stonewu.fusion.service.generation.job.GenerationRequestCommands;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/generation/executions")
@RequiredArgsConstructor
public class GenerationExecutionController {
    private final GenerationJobQueries queries;
    private final GenerationJobCommands commands;
    private final CanvasActorResolver actors;
    private final GenerationRequestCommands requests;

    @GetMapping("/scope")
    public CommonResult<GenerationClientScope> scope() {
        var actor = actors.current();
        return CommonResult.success(new GenerationClientScope(actor.userId(), actor.teamId()));
    }

    @PostMapping("/requests/{requestId}/withdraw")
    public CommonResult<GenerationRequestCommands.Withdrawal> withdraw(@PathVariable String requestId,
                                                                       @Valid @RequestBody WithdrawRequest request) {
        var actor = actors.current();
        request.scope().require(actor);
        return CommonResult.success(requests.withdraw(requestId, request.kind(), actor));
    }

    @GetMapping("/{id}")
    public CommonResult<GenerationJobQueries.View> get(@PathVariable String id) {
        return CommonResult.success(queries.get(id, actors.current()));
    }

    @PostMapping("/jobs/{jobId}/review")
    public CommonResult<Boolean> review(@PathVariable String jobId,
                                        @Valid @RequestBody GenerationJobCommands.Review request) {
        commands.review(jobId, request, actors.current());
        return CommonResult.success(true);
    }

    public record WithdrawRequest(@NotNull GenerationKind kind, @Valid @NotNull GenerationClientScope scope) {
    }
}
