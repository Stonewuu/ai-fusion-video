package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.annotation.Isolation;

import java.util.ArrayList;
import java.util.List;

/** 在同一个数据库读取事务内取得消息和控制状态；事务中不访问 Redis。 */
@Component
@RequiredArgsConstructor
public class AgentStreamHistoryRepository {
    private final AgentRunMapper runs;
    private final AgentMessageMapper messages;
    private final AgentEventRepository controls;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public History read(AgentRun observed) {
        List<AgentRun> members = runs.selectStreamMembers(observed.getStreamRunId());
        boolean nested = !observed.getRunId().equals(observed.getStreamRunId());
        AgentRun current = members.stream().filter(run -> run.getRunId().equals(observed.getRunId()))
                .findFirst().orElseThrow(() -> new IllegalStateException("消息所属运行不存在"));
        List<AgentMessage> saved = messages.selectStreamHistory(observed.getStreamRunId(),
                nested ? observed.getRunId() : null);
        if (nested) {
            saved.forEach(message -> message.setParentToolCallId(null));
        }
        List<Control> committed = new ArrayList<>();
        for (AgentRun member : members) {
            if (nested && !member.getRunId().equals(observed.getRunId())) {
                continue;
            }
            long cursor = 0;
            long through = member.getNextSequence() - 1;
            while (cursor < through) {
                List<CommittedAgentEvent> page = controls.loadReplayPage(member.getRunId(), cursor, through, 256);
                if (page.isEmpty()) {
                    throw new IllegalStateException("运行控制记录不完整");
                }
                page.forEach(event -> committed.add(new Control(member, event)));
                cursor = page.getLast().sequence();
            }
        }
        return new History(current, List.copyOf(saved), List.copyOf(committed));
    }

    public record History(AgentRun run, List<AgentMessage> messages, List<Control> controls) {
    }

    public record Control(AgentRun run, CommittedAgentEvent event) {
    }
}
