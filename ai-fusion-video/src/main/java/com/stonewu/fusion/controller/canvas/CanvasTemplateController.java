package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateBindingResolver.Preview;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Bindings;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Template;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateExport;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateService;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.TemplateImportCommand;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import static com.stonewu.fusion.common.CommonResult.success;

@RestController
@RequiredArgsConstructor
public class CanvasTemplateController {
    private final CanvasTemplateExport exports;
    private final CanvasTemplateService service;
    private final CanvasActorResolver actors;

    public record Input(@NotBlank @Size(max = 16777216) String templateJson, Bindings bindings) {
    }

    @GetMapping("/api/canvases/{canvas}/template")
    public CommonResult<Template> export(@PathVariable String canvas, @RequestParam long expectedRevision) {
        return success(exports.create(canvas, expectedRevision, actors.current()));
    }

    @PostMapping("/api/canvas-templates/inspect")
    public CommonResult<Template> inspect(@Valid @RequestBody Input input) {
        return success(service.inspect(input.templateJson(), actors.current()));
    }

    @PostMapping("/api/canvas-templates/preview")
    public CommonResult<Preview> preview(@Valid @RequestBody Input input) {
        return success(service.preview(input.templateJson(), input.bindings(), actors.current()));
    }

    @PostMapping("/api/canvas-templates/import")
    public CommonResult<ResourceCommandReceipt> create(@Valid @RequestBody TemplateImportCommand input) {
        return success(service.create(input, actors.current()));
    }
}
