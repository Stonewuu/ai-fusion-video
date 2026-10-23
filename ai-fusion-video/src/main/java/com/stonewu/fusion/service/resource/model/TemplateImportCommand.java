package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Bindings;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record TemplateImportCommand(@NotBlank @Pattern(regexp = "[A-Za-z0-9:._-]{1,96}") String requestId,
                                    @NotBlank @Pattern(regexp = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}") String canvasId,
                                    @NotBlank @Size(max = 200) String title,
                                    @NotBlank @Size(max = 16777216) String templateJson,
                                    @NotNull @Valid Bindings bindings,
                                    @NotBlank @Pattern(regexp = "[0-9a-f]{64}") String previewHash) implements ResourceCommand {
    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.IMPORT_TEMPLATE;
    }
}
