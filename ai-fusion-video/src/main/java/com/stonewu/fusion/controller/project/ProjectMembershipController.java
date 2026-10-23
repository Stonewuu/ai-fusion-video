package com.stonewu.fusion.controller.project;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.project.ProjectMembershipData.*;
import com.stonewu.fusion.service.project.ProjectMembershipService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/project/{projectId}/members")
@RequiredArgsConstructor
public class ProjectMembershipController {
    private final ProjectMembershipService service;
    private final CanvasActorResolver actors;

    @GetMapping
    public CommonResult<Members> members(@PathVariable long projectId,
                                         @RequestParam(defaultValue = "1") int page,
                                         @RequestParam(defaultValue = "40") int size) {
        return CommonResult.success(service.members(projectId, actors.current(), page, size));
    }

    @GetMapping("/find")
    public CommonResult<Member> find(@PathVariable long projectId, @RequestParam String username) {
        return CommonResult.success(service.find(projectId, username, actors.current()));
    }

    @GetMapping("/history")
    public CommonResult<PageResult<Audit>> history(@PathVariable long projectId,
                                                   @RequestParam(defaultValue = "1") int page,
                                                   @RequestParam(defaultValue = "40") int size) {
        return CommonResult.success(service.history(projectId, actors.current(), page, size));
    }

    @PostMapping
    public CommonResult<Receipt> change(@PathVariable long projectId, @Valid @RequestBody Change command) {
        return CommonResult.success(service.change(projectId, command, actors.current()));
    }
}
