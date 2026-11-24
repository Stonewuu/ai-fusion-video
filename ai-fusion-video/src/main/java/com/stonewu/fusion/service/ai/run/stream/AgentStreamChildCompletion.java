package com.stonewu.fusion.service.ai.run.stream;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/** 子任务完成以数据库中的终态和完整消息为准，不再依赖文本片段回放。 */
@Component
@RequiredArgsConstructor
public class AgentStreamChildCompletion {
    private final AgentRunMapper runs;
    private final AgentMessageMapper messages;
    private final AgentRuntimeSchedulers schedulers;

    public Mono<PlatformSubAgentRun> await(PlatformSubAgentRun child) {
        return Mono.fromCallable(() -> runs.selectByRunId(child.childRunId())).subscribeOn(schedulers.journal())
                .repeatWhen(repeats -> repeats.delayElements(Duration.ofMillis(200)))
                .filter(run -> AgentRunStatus.valueOf(run.getStatus()).isTerminal()).next()
                .flatMap(run -> Mono.fromCallable(() -> {
                    List<AgentMessage> output = messages.selectList(new LambdaQueryWrapper<AgentMessage>()
                            .eq(AgentMessage::getRunId, child.childRunId()).eq(AgentMessage::getRole, "assistant")
                            .orderByAsc(AgentMessage::getMessageOrder));
                    String text = output.stream().map(AgentMessage::getContent).filter(Objects::nonNull)
                            .collect(Collectors.joining());
                    AgentRunStatus status = AgentRunStatus.valueOf(run.getStatus());
                    return new PlatformSubAgentRun(child.childRunId(), child.parentRunId(), child.parentToolCallId(),
                            child.agentName(), status, text.isBlank() ? null : text, run.getErrorMessage(), 2);
                }).subscribeOn(schedulers.journal()));
    }
}
