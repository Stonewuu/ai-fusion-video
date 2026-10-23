package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.run.*;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.model.ResumeAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.ResumeExternalCommand;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import io.agentscope.core.message.Msg;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

/**
 * 从可信数据库状态恢复等待；无浏览器依赖，不接受客户端伪造的工具执行结果。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class CanvasAgentWaitBridge {
    private final CanvasChildAwait children;
    private final CanvasAgentWaitRepository queue;
    private final CanvasAgentScopeService scopes;
    private final ToolExecutorRegistry tools;
    private final AiModelService models;
    private final AgentWaitingStatePort waiting;
    private final AgentExecutionRuntimeContextRequests runtimeContexts;
    private final RunExecutionSupervisor supervisor;
    private final AgentKernelSnapshotBuilder snapshots;
    private final AgentRuntimeInstanceIdentity identity;
    private final AgentScopeV2Properties properties;
    private final CanvasJson json;
    private final ObjectMapper mapper;
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(4);
    private final Set<String> active = ConcurrentHashMap.newKeySet();
    private static final Duration STEP_TIMEOUT = Duration.ofSeconds(30);

    @Scheduled(fixedDelay = 1000)
    public void tick() {
        for (String id : queue.waiting()) {
            submit("wait:" + id, () -> check(id));
        }
        for (String id : queue.dispatching(identity.value())) {
            submit("resume:" + id, () -> dispatch(id));
        }
    }

    private void submit(String key, Runnable action) {
        if (!slots.tryAcquire()) {
            return;
        }
        if (!active.add(key)) {
            slots.release();
            return;
        }
        workers.submit(() -> {
            try {
                action.run();
            } catch (BusinessException failure) {
                if (failure.getCode() != 409) {
                    log.warn("画布助手恢复暂未完成 [{}]: {}", key, failure.getMessage());
                }
            } catch (RuntimeException failure) {
                log.warn("画布助手恢复步骤失败 [{}]", key, failure);
            } finally {
                active.remove(key);
                slots.release();
            }
        });
    }

    public void check(String id) {
        AgentRun agent = queue.deferCheck(id);
        if (agent == null || !"WAITING_EXTERNAL".equals(agent.getStatus())) {
            return;
        }
        var scope = scopes.decode(agent.getCanvasScopeJson());
        var pending = waiting.getPendingExternalAuthorized(id, agent.getUserId(), agent.getWaitingToolCallId())
                .block(STEP_TIMEOUT);
        if (pending == null) {
            throw new IllegalStateException("外部等待记录不存在");
        }
        var payload = json.read(pending.suspendedPayloadJson(), ObjectNode.class);
        var kernel = json.read(agent.getAgentDefinitionSnapshotJson(), AgentKernelSnapshotPayload.class);
        var model = models.getById(Long.parseLong(kernel.modelConfigId()));
        JsonNode calls = payload.path("toolCalls");
        if (!calls.isArray() || calls.isEmpty() || calls.size() > 16) {
            throw new IllegalStateException("等待调用列表无效");
        }
        var result = json.object(Map.of());
        var results = result.putArray("results");
        Set<String> ids = new HashSet<>();
        boolean ready = true;
        for (JsonNode call : calls) {
            String toolId = call.path("id").asText();
            if (toolId.isBlank() || !ids.add(toolId)) {
                throw new IllegalStateException("外部等待工具身份无效");
            }
            String name = call.path("name").asText();
            var executor = tools.findExecutor(name);
            if ((!(executor instanceof CanvasExternalTool) && !CanvasAgentNodeTypes.CHILD_TOOLS.contains(name))
                    || kernel.tools().stream().noneMatch(item -> item.name().equals(name))) {
                throw new IllegalStateException("外部等待工具未登记或不在本轮白名单内");
            }
            var entry = results.addObject().put("toolCallId", toolId).put("toolName", name);
            try {
                var actor = new CanvasActor(agent.getUserId(), agent.getActorTeamId());
                var output = executor instanceof CanvasExternalTool external
                        ? external.poll(json.object(call.path("input")), scope, actor, model)
                        : children.poll(id, toolId, name, scope, actor);
                ready &= output != null;
                if (output != null) {
                    entry.set("result", json.object(output));
                }
            } catch (BusinessException unavailable) {
                if (!Set.of(400, 403, 404, 409, 410, 422).contains(unavailable.getCode())) {
                    throw unavailable;
                }
                entry.set("result", json.object(
                        Map.of("status", "error", "code", unavailable.getCode(), "message", unavailable.getMessage())));
            }
        }
        if (!ready) {
            return;
        }
        waiting.resumeExternal(
                        new ResumeExternalCommand(id, agent.getUserId(), "canvas-run-waiter", pending.toolCallId(),
                                json.write(result), identity.value(), properties.getExecution().getOwnerLease()))
                .block(STEP_TIMEOUT);
        // 恢复消息与 RUNNING 转换同事务保存；即使此处进程退出，下一轮仍会读取恢复发件箱。
    }

    public void dispatch(String id) {
        var dispatch = queue.claim(id, identity.value());
        if (dispatch == null) {
            return;
        }
        var resumed = dispatch.run();
        var payload = json.read(resumed.agentDefinitionSnapshotJson(), AgentKernelSnapshotPayload.class);
        var snapshot = snapshots.build(payload);
        if (!snapshot.fingerprint().equals(resumed.kernelFingerprint())) {
            throw new IllegalStateException("恢复配置摘要不一致");
        }
        var saved = json.read(dispatch.resultJson(), ObjectNode.class);
        List<Msg> messages;
        if (saved.has("confirmation")) {
            var confirmation = saved.path("confirmation");
            var decisions = new LinkedHashMap<String, Boolean>();
            confirmation.path("decisions").fields().forEachRemaining(entry -> {
                if (!entry.getValue().isBoolean()) {
                    throw new IllegalStateException("持久化审批决定无效");
                }
                decisions.put(entry.getKey(), entry.getValue().booleanValue());
            });
            messages = List.of(
                    AgentConfirmationMessages.restore(confirmation.path("toolCalls").toString(), decisions, mapper));
        } else {
            messages = CanvasAgentResumeMessages.external(saved, tools);
        }
        var permission = ToolExecutionMode.parse(
                payload.promptVariables().get(AgentKernelSpecFactory.TOOL_EXECUTION_MODE_VARIABLE));
        runtimeContexts.forResume(resumed, payload.agentDefinitionStableKey(), permission)
                .flatMap(runtime -> supervisor.resume(new ResumeAgentExecutionCommand(resumed,
                        messages, snapshot, runtime)))
                .onErrorResume(OwnedExecutionRegistry.ExecutionAlreadyOwnedException.class,
                        alreadyRunning -> Mono.empty())
                .block(STEP_TIMEOUT);
        queue.clear(id, resumed.newOwnerEpoch(), dispatch.sequence());
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
