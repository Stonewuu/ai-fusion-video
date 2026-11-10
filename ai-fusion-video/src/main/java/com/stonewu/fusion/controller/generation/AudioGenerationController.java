package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.generation.audio.AudioTaskData;
import com.stonewu.fusion.service.generation.audio.AudioTaskService;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/generation/audio")
@RequiredArgsConstructor
public class AudioGenerationController {
    private final AudioTaskService tasks;
    private final CanvasActorResolver actors;

    @PostMapping("/submit")
    public CommonResult<AudioTaskData.View> create(@Valid @RequestBody AudioTaskData.Create request) {
        return CommonResult.success(tasks.create(request, actors.current()));
    }

    @GetMapping("/{id}")
    public CommonResult<AudioTaskData.View> get(@PathVariable String id) {
        return CommonResult.success(tasks.get(id, actors.current()));
    }

    @PostMapping("/{id}/retry-save")
    public CommonResult<Boolean> retrySave(@PathVariable String id) {
        tasks.retrySave(id, actors.current());
        return CommonResult.success(true);
    }

    @GetMapping
    public CommonResult<List<AudioTaskData.View>> list() {
        return CommonResult.success(tasks.list(actors.current()));
    }
}
