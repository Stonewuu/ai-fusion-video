package com.stonewu.fusion.service.canvas.template;

import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.TemplateImportCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Bindings;
import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Template;

@Service
@RequiredArgsConstructor
public class CanvasTemplateService {
    private final CanvasTemplateCodec codec;
    private final CanvasTemplateBindingResolver bindings;
    private final ResourceCommandService commands;
    private final CanvasAccessPolicy access;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Template inspect(String content, CanvasActor actor) {
        access.requireActorScope(actor);
        return codec.parse(content);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public CanvasTemplateBindingResolver.Preview preview(String content, Bindings choices, CanvasActor actor) {
        access.requireActorScope(actor);
        return bindings.preview(bindings.resolve(codec.parse(content), choices, actor));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", allEntries = true)
    public ResourceCommandReceipt create(TemplateImportCommand command, CanvasActor actor) {
        return commands.execute(command, ResourceCommandContext.user(actor, null));
    }
}
