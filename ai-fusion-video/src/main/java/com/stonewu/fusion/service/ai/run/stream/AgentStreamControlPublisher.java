package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.AgentMessageProjectionService;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;

/** 把已经提交的低频控制状态放入展示流，多次读取同一状态不重复发布。 */
@Component
@RequiredArgsConstructor
public class AgentStreamControlPublisher {
    private final AgentRunMapper runs;
    private final AgentEventRepository controls;
    private final AgentMessageProjectionService projections;
    private final AgentRedisStreamStore streams;
    private final AgentStreamEventFormatter formatter;
    private final AgentRuntimeSchedulers schedulers;
    private final AgentStreamRetention retention;

    public Mono<Void> synchronize(String streamRunId) {
        return Mono.fromCallable(() -> runs.selectStreamMembers(streamRunId)).subscribeOn(schedulers.journal())
                .flatMap(members -> {
                    boolean ended = !members.isEmpty() && members.stream().allMatch(run -> run.getTerminalSequence() != null);
                    return Mono.fromCallable(() -> !ended || streams.exists(streamRunId))
                            .subscribeOn(schedulers.journal()).flatMap(retained -> {
                                if (!retained) {
                                    return Mono.empty();
                                }
                                return Flux.fromIterable(members).concatMap(run -> publishRun(run, ended))
                                        .then(ended ? retention.applyIfFinished(streamRunId) : Mono.empty());
                            });
                });
    }

    private Mono<Void> publishRun(AgentRun run, boolean ended) {
        return Mono.fromCallable(() -> {
            long after = streams.controlSequence(run.getStreamRunId(), run.getRunId());
            long through = run.getNextSequence() - 1;
            return after >= through ? List.<CommittedAgentEvent>of()
                    : controls.loadReplayPage(run.getRunId(), after, through, 256);
        }).subscribeOn(schedulers.journal()).flatMapMany(Flux::fromIterable)
                .concatMap(event -> {
                    boolean terminal = run.getTerminalSequence() != null
                            && event.sequence() == run.getTerminalSequence();
                    boolean projectMessages = terminal || "USER_CONFIRM_RESULT".equals(event.outputType());
                    return (projectMessages ? projections.projectThrough(run.getRunId(), event.sequence()) : Mono.<Void>empty())
                            .then(Mono.fromRunnable(() -> publish(run, event, ended)).subscribeOn(schedulers.journal()));
                }).then();
    }

    private void publish(AgentRun run, CommittedAgentEvent event, boolean ended) {
        if (event.publishRequired()) {
            AiChatStreamRespVO response = formatter.format(run, event.envelope());
            if (ended) {
                streams.appendExisting(run.getStreamRunId(), identity(run.getRunId(), event.sequence()), response);
            } else {
                streams.append(run.getStreamRunId(), identity(run.getRunId(), event.sequence()), response);
            }
        }
        streams.advanceControl(run.getStreamRunId(), run.getRunId(), event.sequence());
    }

    public static String identity(String sourceRunId, long sequence) {
        return "control:" + sourceRunId + ':' + sequence;
    }
}
