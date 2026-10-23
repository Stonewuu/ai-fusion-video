package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.ResourceCommandRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.*;

@Service
public class ResourceCommandService {
    private record HandlerKey(CanvasValueType resourceType, ResourceOperation operation) {
    }

    private final Map<HandlerKey, ResourceCommandHandler> handlers = new HashMap<>();
    private final ResourceCommandRepository commands;
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final CanvasRequestValidator validation;
    private final CanvasJson json;
    private final ResourceAgentFence agentFence;
    private final TransactionTemplate writes;
    private final List<ResourceMutationObserver> observers;

    public ResourceCommandService(List<ResourceCommandHandler> handlers, ResourceCommandRepository commands,
                                  CanvasDocumentRepository canvases, CanvasAccessPolicy access,
                                  CanvasRequestValidator validation,
                                  CanvasJson json, ResourceAgentFence agentFence, PlatformTransactionManager manager,
                                  List<ResourceMutationObserver> observers) {
        for (ResourceCommandHandler handler : handlers) {
            for (CanvasValueType type : handler.resourceTypes()) {
                if (this.handlers.putIfAbsent(new HandlerKey(type, handler.operation()), handler) != null) {
                    throw new IllegalStateException("业务写入处理器重复");
                }
            }
        }
        this.commands = commands;
        this.canvases = canvases;
        this.access = access;
        this.validation = validation;
        this.json = json;
        this.agentFence = agentFence;
        this.observers = List.copyOf(observers);
        this.writes = new TransactionTemplate(manager);
        writes.setIsolationLevel(TransactionDefinition.ISOLATION_READ_COMMITTED);
    }

    public ResourceCommandReceipt patch(ResourcePatchCommand command, ResourceCommandContext context) {
        return execute(command, context);
    }

    public ResourceCommandReceipt execute(ResourceCommand command, ResourceCommandContext context) {
        validation.validate(command);
        Integer parentVersion = switch (command) {
            case ResourcePatchCommand patch -> patch.expectedParentVersion();
            case ResourceDeleteCommand deletion -> deletion.expectedParentVersion();
            default -> null;
        };
        if (parentVersion != null && command.resourceType() != CanvasValueType.ASSET_ITEM) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "此命令不接收父级资产版本");
        }
        if (command instanceof ResourcePatchCommand patch && patch.patch().isEmpty()) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "修改字段不能为空");
        }
        ResourceCommandHandler handler = handlers.get(new HandlerKey(command.resourceType(), command.operation()));
        if (handler == null) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "此业务类型未登记该命令");
        }
        var binding = json.object(context);
        // 重启后的 owner epoch 可以变化，但同一次工具调用的已提交命令仍应返回原回执。
        binding.remove(List.of("ownerInstanceId", "ownerEpoch"));
        String hash = json.hash(Map.of("operation", command.operation(), "command", command, "context", binding));
        ResourceCommandReceipt result = Objects.requireNonNull(writes.execute(ignored -> {
            String commandId = UUID.randomUUID().toString();
            var admission = commands.admit(commandId, command, context, hash);
            boolean matching = admission.hash().equals(hash);
            // 重放也必须先重新授权，不允许通过旧 requestId 读取已撤销权限的数据。
            Long projectId = handler.authorize(command, context, matching ? admission.receipt() : null);
            requireWriteScope(command, projectId, context);
            agentFence.requireActive(context, projectId);
            if (context.canvasId() != null) {
                validation.id(context.canvasId());
                CanvasDocument canvas = canvases.require(context.canvasId(), true);
                access.requireEdit(canvas, context.actor());
                if (!Objects.equals(canvas.projectId(), projectId)) {
                    throw new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "业务对象不属于当前画布项目",
                            Map.of());
                }
            }
            if (!matching) {
                throw new CanvasException(409, CanvasErrorCode.REQUEST_CONFLICT,
                        "同一请求 ID 不能提交不同的业务命令", Map.of());
            }
            if (admission.receipt() != null) {
                return admission.receipt();
            }
            if (!admission.id().equals(commandId)) {
                throw new IllegalStateException("业务命令存在未提交的持久化记录");
            }
            observers.forEach(observer -> observer.before(command, context));
            ResourceMutation mutation = handler.apply(command, context);
            observers.forEach(observer -> observer.validateAndApply(mutation, context));
            requireWriteScope(command, projectId, context);
            agentFence.requireActive(context, projectId);
            ObjectNode response = json.read(json.write(mutation.result()), ObjectNode.class);
            ResourceCommandReceipt receipt = new ResourceCommandReceipt(commandId, command.requestId(), response,
                    mutation.changes());
            commands.commit(receipt, mutation);
            return receipt;
        }));
        // 事务提交后清理对应业务缓存；处理器的 evict 由 Spring Cache 代理执行。
        handler.evict();
        return result;
    }

    private void requireWriteScope(ResourceCommand command, Long projectId, ResourceCommandContext context) {
        access.requireActorScope(context.actor());
        // 项目删除由处理器检查管理权；删除后的最小幂等回执不再查询已删除的项目。
        if (projectId != null && !(command.resourceType() == CanvasValueType.PROJECT
                && (command.operation() == ResourceOperation.DELETE || command.operation() == ResourceOperation.RESTORE))) {
            access.requireProjectEditor(projectId, context.actor());
        }
    }
}
