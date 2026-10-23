package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.config.ArtStylePresets;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaPresetPublisher;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.io.ByteArrayInputStream;
import java.io.IOException;

@RestController
@RequestMapping("/api/media-presets")
@RequiredArgsConstructor
public class MediaPresetController {
    private final CanvasActorResolver actors;
    private final PresetArtStyleResourceResolver files;
    private final MediaIntakeService intake;
    private final MediaPresetPublisher publisher;

    public record Request(@NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId) {
    }

    @PostMapping("/{key}/publish")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<String> publish(@PathVariable String key,
                                        @Valid @RequestBody Request request) throws IOException {
        var preset = ArtStylePresets.getByKey(key);
        if (preset == null) {
            throw MediaException.invalid("内置画风不存在");
        }
        var source = files.load(preset.getReferenceImagePath());
        var context = ResourceCommandContext.user(actors.current(), null);
        String media;
        try (var input = new ByteArrayInputStream(source.bytes())) {
            media = intake.upload(input, source.fileName(), source.mimeType(), request.requestId(), null, context).id();
        }
        return CommonResult.success(publisher.publish(key, media, context));
    }
}
