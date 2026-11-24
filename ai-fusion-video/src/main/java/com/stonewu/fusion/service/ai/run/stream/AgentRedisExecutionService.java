package com.stonewu.fusion.service.ai.run.stream;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

@Component
@RequiredArgsConstructor
public class AgentRedisExecutionService {
    private final AgentRunMapper runs;
    private final AgentRedisStreamStore streams;
    private final AgentStreamMessageRepository messages;
    private final AgentStreamEventFormatter formatter;
    private final AgentEventRepository controls;
    private final AgentStreamControlPublisher controlPublisher;
    private final ObjectMapper json;
    private final AgentRuntimeSchedulers schedulers;
    private final ConcurrentHashMap<String, AgentStreamSession> sessions = new ConcurrentHashMap<>();

    public Mono<Void> begin(AgentRunContext context) {
        return controlPublisher.synchronize(context.streamRunId()).then(Mono.fromRunnable(() -> {
            AgentRun run = runs.selectByRunId(context.runId());
            if (run == null || run.getStreamProtocolVersion() != 2) {
                throw new IllegalStateException("运行未使用 Redis 消息协议");
            }
            AgentStreamSession session = new AgentStreamSession(run, context.ownerInstanceId(), context.ownerEpoch(),
                    streams, messages, formatter, controls, json);
            if (sessions.putIfAbsent(context.runId(), session) != null) {
                throw new IllegalStateException("消息流已在当前实例执行");
            }
        }).subscribeOn(schedulers.journal())).then();
    }

    public Mono<Optional<CommittedAgentEvent>> accept(String runId, AgentEventEnvelope event) {
        return Mono.fromCallable(() -> session(runId).accept(event)).subscribeOn(schedulers.journal());
    }

    public Mono<Void> finish(String runId) {
        return Mono.fromRunnable(() -> {
            AgentStreamSession session = sessions.get(runId);
            if (session != null) {
                session.finish();
            }
        }).subscribeOn(schedulers.journal()).then();
    }

    public void release(String runId) {
        sessions.remove(runId);
    }

    public Mono<Void> afterTerminal(String runId) {
        return Mono.fromCallable(() -> runs.selectByRunId(runId)).subscribeOn(schedulers.journal())
                .flatMap(run -> run.getStreamProtocolVersion() == 2
                        ? controlPublisher.synchronize(run.getStreamRunId()) : Mono.empty());
    }

    public AgentStreamToolBarrier barrier(String runId) {
        return new AgentStreamToolBarrier(ids -> awaitCalls(runId, ids), ids -> awaitResults(runId, ids));
    }

    private Mono<Void> awaitCalls(String runId, List<String> ids) {
        return Flux.fromIterable(ids).concatMap(id -> Mono.fromCallable(() -> messages.hasToolCall(runId, id))
                .subscribeOn(schedulers.journal())
                .flatMap(saved -> saved ? Mono.empty() : session(runId).awaitCall(id))).then();
    }

    private Mono<Void> awaitResults(String runId, List<String> ids) {
        return Flux.fromIterable(ids).concatMap(id -> session(runId).awaitResult(id)).then();
    }

    private AgentStreamSession session(String runId) {
        AgentStreamSession session = sessions.get(runId);
        if (session == null) {
            throw new IllegalStateException("运行的消息流未启动");
        }
        return session;
    }
}
