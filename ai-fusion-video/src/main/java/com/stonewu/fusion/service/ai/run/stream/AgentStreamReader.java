package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.controller.ai.vo.PipelineRunStatusRespVO;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Component
@RequiredArgsConstructor
public class AgentStreamReader {
    private final AgentRedisStreamStore streams;
    private final AgentStreamControlPublisher controls;
    private final AgentRuntimeSchedulers schedulers;
    private final AgentStreamHistoryRepository history;
    private final AgentStreamEventFormatter formatter;
    private final AgentEventRepository events;

    /** 每次连接先重建完整历史和未完成草稿，读取位置只用于随后到达的增量。 */
    public Flux<AiChatStreamRespVO> read(AgentRun run, long afterSequence) {
        return controls.synchronize(run.getStreamRunId()).then(Mono.fromCallable(() -> {
            // 先取得 Redis 位置，再读数据库；提交已发生但尚未发布的事件由快照身份去重。
            long through = streams.lastSequence(run.getStreamRunId());
            AgentStreamSnapshot snapshot = new AgentStreamSnapshot(history.read(run), through);
            if (!snapshot.terminal() && afterSequence > through) {
                throw new BusinessException(400, "消息流读取位置无效，请重新打开会话");
            }
            var drafts = new ArrayList<AiChatStreamRespVO>();
            long cursor = 0;
            while (!snapshot.terminal() && cursor < through) {
                var page = streams.readAfter(run.getStreamRunId(), cursor, 256);
                if (page.isEmpty()) {
                    throw new IllegalStateException("运行中的消息流不完整");
                }
                for (var event : page) {
                    if (event.getSequence() <= through && visible(run, event) && snapshot.includes(event)) {
                        drafts.add(observed(run, event));
                    }
                }
                cursor = page.getLast().getSequence();
            }
            return new Bootstrap(snapshot, snapshot.response(drafts, formatter), through);
        }).subscribeOn(schedulers.journal())).flatMapMany(bootstrap -> Flux.just(bootstrap.response())
                .concatWith(bootstrap.snapshot().terminal() ? Flux.empty() : follow(run, bootstrap)));
    }

    private Flux<AiChatStreamRespVO> follow(AgentRun run, Bootstrap bootstrap) {
        AtomicLong cursor = new AtomicLong(bootstrap.cursor());
        AtomicBoolean done = new AtomicBoolean();
        return Flux.defer(() -> controls.synchronize(run.getStreamRunId())
                        .then(Mono.fromCallable(() -> streams.readAfter(run.getStreamRunId(), cursor.get(), 256))
                                .subscribeOn(schedulers.journal()))
                        .flatMapMany(page -> page.isEmpty()
                                ? Mono.delay(Duration.ofMillis(100)).thenMany(Flux.empty()) : Flux.fromIterable(page))
                        .doOnNext(event -> cursor.set(event.getSequence()))
                        .filter(event -> visible(run, event) && bootstrap.snapshot().includes(event))
                        .map(event -> observed(run, event))
                        .doOnNext(event -> {
                            if (terminal(event)) {
                                done.set(true);
                            }
                        }).takeUntil(AgentStreamReader::terminal))
                .repeat(() -> !done.get());
    }

    /** 状态接口只读取 MySQL 状态与 Redis 位置，不同步或重新创建展示流。 */
    public Mono<PipelineRunStatusRespVO> status(AgentRun run) {
        return Mono.fromCallable(() -> {
            long sequence = streams.lastSequence(run.getStreamRunId());
            AiChatStreamRespVO terminal = null;
            if (run.getTerminalSequence() != null) {
                var committed = events.loadReplayPage(run.getRunId(), run.getTerminalSequence() - 1,
                        run.getTerminalSequence(), 1);
                if (committed.size() != 1) {
                    throw new IllegalStateException("运行结束记录不存在");
                }
                terminal = observed(run, formatter.format(run, committed.getFirst().envelope())).setSequence(sequence);
            }
            return new PipelineRunStatusRespVO(run.getRunId(), run.getStatus(), sequence,
                    run.getWaitingReplyId(), terminal);
        }).subscribeOn(schedulers.journal());
    }

    private static boolean visible(AgentRun run, AiChatStreamRespVO event) {
        return run.getRunId().equals(run.getStreamRunId()) || run.getRunId().equals(event.getSourceRunId());
    }

    static AiChatStreamRespVO observed(AgentRun run, AiChatStreamRespVO event) {
        if (!run.getRunId().equals(run.getStreamRunId())) {
            event.setRunId(run.getRunId()).setParentToolCallId(null).setAgentName(null);
        }
        return event;
    }

    private static boolean terminal(AiChatStreamRespVO event) {
        return event.getParentToolCallId() == null && event.getAgentName() == null
                && ("DONE".equals(event.getOutputType()) || "ERROR".equals(event.getOutputType())
                || "CANCELLED".equals(event.getOutputType()));
    }

    private record Bootstrap(AgentStreamSnapshot snapshot, AiChatStreamRespVO response, long cursor) {
    }
}
