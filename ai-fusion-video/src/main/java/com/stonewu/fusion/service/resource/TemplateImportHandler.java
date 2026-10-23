package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateBindingResolver;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateCodec;
import com.stonewu.fusion.service.canvas.template.CanvasTemplateWriter;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class TemplateImportHandler implements ResourceCommandHandler {
    private final CanvasAccessPolicy access;
    private final CanvasDocumentRepository canvases;
    private final ResourceReadRepository records;
    private final CanvasTemplateCodec codec;
    private final CanvasTemplateBindingResolver bindings;
    private final ProjectContentCopies copies;
    private final CanvasTemplateWriter writer;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.IMPORT_TEMPLATE;
    }

    @Override
    public Long authorize(ResourceCommand command, ResourceCommandContext context, ResourceCommandReceipt previous) {
        if (context.canvasId() != null || !context.origin().equals("user")) {
            throw new BusinessException(400, "模板导入必须由明确的导入操作提交");
        }
        access.requireActorScope(context.actor());
        if (previous != null) {
            access.requireProject(previous.result().path("id").longValue(), context.actor());
            access.requireView(canvases.require(previous.result().path("canvasId").asText(), false), context.actor());
        }
        return null;
    }

    @Override
    public ResourceMutation apply(ResourceCommand command, ResourceCommandContext context) {
        var input = (TemplateImportCommand) command;
        if (canvases.find(input.canvasId(), true).isPresent()) {
            throw new BusinessException(409, "目标画布编号已用于其他操作");
        }
        var template = codec.parse(input.templateJson());
        var resolved = bindings.resolve(template, input.bindings(), context.actor());
        resolved.media().values().stream().map(file -> file.projectId()).filter(Objects::nonNull).distinct().sorted()
                .forEach(project -> records.require(CanvasValueType.PROJECT, project, true));
        // 等待来源权限锁后再次读取；不能用等锁前的授权继续复制已撤权素材。
        resolved = bindings.resolve(template, input.bindings(), context.actor());
        if (!bindings.preview(resolved).hash().equals(input.previewHash())) {
            throw new BusinessException(409, "模板或所选绑定在预览后变化，请重新检查");
        }
        var copied = copies.create(input.title(), input.canvasId(), template.graph().projectId(),
                template.primaryScript(), template.primaryStoryboard(),
                template.objects(), resolved.media(), context.actor());
        for (var change : copied.changes()) {
            try {
                var header = records.historical(change.resourceType(), change.resourceId());
                if (!Objects.equals(header.projectId(), copied.projectId())) {
                    throw new BusinessException(400, "模板父子归属不一致");
                }
            } catch (BusinessException failure) {
                if (failure.getCode() == 404) {
                    throw new BusinessException(400, "模板父子关系不一致，导入未提交");
                }
                throw failure;
            }
        }
        var canvas = writer.write(input, resolved, copied, context.actor());
        var result = json.object(
                Map.of("id", copied.projectId(), "canvasId", canvas.id(), "title", input.title(), "templateHash",
                        json.hash(template),
                        "businessObjects", copied.changes().size(), "mediaFiles", copied.media().size()));
        var objectMap = copied.objects().entrySet().stream()
                .map(entry -> Map.of("type", entry.getKey().type(), "templateId", entry.getKey().id(), "id",
                        entry.getValue())).toList();
        var mediaMap = copied.media().entrySet().stream()
                .map(entry -> Map.of("slot", entry.getKey(), "id", entry.getValue().id())).toList();
        return new ResourceMutation(result, copied.changes(),
                json.object(Map.of("created", copied.changes(), "template", Map.of(
                        "hash", json.hash(template), "objects", objectMap, "media", mediaMap, "bindings",
                        input.bindings()))));
    }

    @Override
    @CacheEvict(value = {"project", "script", "storyboard", "asset", "resourceDocument", "canvasGraph"}, allEntries = true)
    public void evict() {
    }
}
