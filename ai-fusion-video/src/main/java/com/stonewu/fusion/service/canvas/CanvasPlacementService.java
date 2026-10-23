package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.CreateNode;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

/**
 * 新建业务对象并放上画布在同一事务完成；任一版本/图约束失败都不留下半成品。
 */
@Service
@RequiredArgsConstructor
public class CanvasPlacementService {
    public record CreateAndPlace(
            @NotBlank @Size(max = 96) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
            @NotBlank String nodeId, @NotBlank @Size(max = 200) String title,
            @NotNull CanvasValueType resourceType, @NotNull ObjectNode fields, double x, double y) {
    }

    public record Result(ResourceCommandReceipt resource, CanvasCommandReceipt canvas) {
    }

    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasRequestValidator requests;
    private final CanvasJson json;
    private final ResourceCommandService resources;
    private final CanvasCommandService canvases;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.canvas().revision() - 1)")
    public Result create(String canvasId, CreateAndPlace command, CanvasActor actor) {
        return create(canvasId, command, ResourceCommandContext.user(actor, canvasId));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", key = "#canvasId + ':' + (#result.canvas().revision() - 1)")
    public Result create(String canvasId, CreateAndPlace command, ResourceCommandContext context) {
        CanvasActor actor = context.actor();
        requests.validate(command);
        requests.id(canvasId);
        requests.id(command.nodeId());
        if (!command.resourceType().isBusinessResource()) {
            throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "该类型不是可创建的业务对象");
        }
        var document = documents.require(canvasId, false);
        access.requireEdit(document, actor);
        var resource = resources.execute(
                new ResourceCreateCommand(command.requestId() + ":resource", command.resourceType(),
                        document.projectId(), command.fields()),
                context);
        long id = resource.result().path("id").longValue();
        long version = resource.result().path("version").longValue();
        if (id <= 0 || !resource.result().path("version").isIntegralNumber()) {
            throw new IllegalStateException("业务创建没有返回对象及版本");
        }
        var node = new CanvasNode(command.nodeId(), CanvasNodeKind.fromCode(command.resourceType().code()), 1,
                command.title(),
                new CanvasNodeLayout(command.x(), command.y(), 300.0, null, null, false, false), json.object(Map.of()),
                new CanvasResourceReference(command.resourceType(), id, version), 0, 0);
        var canvas = canvases.edit(canvasId,
                new CanvasEditCommand(command.requestId() + ":node", List.of(new CreateNode(node))), context);
        return new Result(resource, canvas);
    }
}
