package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;
import reactor.core.publisher.Mono;

import java.time.ZoneOffset;
import java.util.Comparator;

@Component
@RequiredArgsConstructor
@Slf4j
public class AgentStreamRetention {
    private final AgentRunMapper runs;
    private final AgentRedisStreamStore streams;
    private final AgentRuntimeSchedulers schedulers;
    private final AgentScopeV2Properties properties;

    /** 独立于被取消的执行订阅；数据库事务提交后才在调度线程访问 Redis。 */
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void afterMessagesCommitted(AgentStreamMessagesCommitted event) {
        applyIfFinished(event.streamRunId()).subscribe(ignored -> {
        }, failure -> log.error("设置消息流保留期限失败：streamRunId={}", event.streamRunId(), failure));
    }

    public Mono<Void> applyIfFinished(String streamRunId) {
        return Mono.fromRunnable(() -> {
            var members = runs.selectStreamMembers(streamRunId);
            if (members.isEmpty() || members.stream().anyMatch(run -> run.getTerminalSequence() == null
                    || run.getProjectionCompletedAt() == null)) {
                return;
            }
            var finished = members.stream().map(AgentRun::getFinishedAt)
                    .max(Comparator.naturalOrder()).orElseThrow();
            streams.expireAt(streamRunId, finished.toInstant(ZoneOffset.UTC).plus(properties.getStreamRetention()));
        }).subscribeOn(schedulers.journal()).then();
    }
}
