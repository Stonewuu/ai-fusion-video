package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationData.Grant;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationData.Member;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationService;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/canvases/{canvasId}/members")
@RequiredArgsConstructor
public class CanvasCollaborationController {
    private final CanvasActorResolver actors;
    private final CanvasCollaborationService collaboration;

    @GetMapping
    public CommonResult<PageResult<Member>> list(@PathVariable String canvasId,
                                                 @RequestParam(defaultValue = "1") int page,
                                                 @RequestParam(defaultValue = "40") int size) {
        return CommonResult.success(collaboration.members(canvasId, actors.current(), page, size));
    }

    @GetMapping("/find")
    public CommonResult<Member> find(@PathVariable String canvasId, @RequestParam String username) {
        return CommonResult.success(collaboration.findMember(canvasId, username, actors.current()));
    }

    @PostMapping
    public CommonResult<CanvasCommandReceipt> grant(@PathVariable String canvasId, @Valid @RequestBody Grant command) {
        return CommonResult.success(collaboration.grant(canvasId, command, actors.current()));
    }
}
