package com.stonewu.fusion.service.ai.text;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.tool.ReadToolResultToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.ReadScriptContentToolExecutor;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.middleware.MiddlewareBase;
import io.agentscope.core.middleware.ReasoningInput;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public final class LongToolResultMiddleware implements MiddlewareBase {
    private final LongToolResultStore results;
    private final int maxInlineChars;
    private final ObjectMapper json = new ObjectMapper();

    public LongToolResultMiddleware(LongToolResultStore results, Integer contextWindow) {
        this.results = results;
        this.maxInlineChars = ToolTextBudget.inlineChars(contextWindow);
    }

    @Override
    public Flux<AgentEvent> onReasoning(Agent agent, RuntimeContext context, ReasoningInput input,
                                       Function<ReasoningInput, Flux<AgentEvent>> next) {
        return Mono.fromCallable(() -> project(agent, context, input))
                .subscribeOn(Schedulers.boundedElastic()).flatMapMany(next);
    }

    private ReasoningInput project(Agent agent, RuntimeContext context, ReasoningInput input) {
        var state = RuntimeContext.resolveAgentState(context, agent);
        var saved = new HashMap<String, String>();
        // 先保存全部长结果，全部成功后才改会话状态，存储失败不产生不可读取的占位内容。
        List<Msg> canonical = state == null ? List.of() : project(state.contextMutable(), context, saved);
        List<Msg> messages = project(input.messages(), context, saved);
        if (state != null) {
            state.contextMutable().clear();
            state.contextMutable().addAll(canonical);
        }
        return new ReasoningInput(messages, input.tools(), input.options());
    }

    private List<Msg> project(List<Msg> messages, RuntimeContext context, Map<String, String> saved) {
        return messages.stream().map(message -> {
            List<ContentBlock> blocks = message.getContent().stream()
                    .map(block -> block instanceof ToolResultBlock result ? project(result, context, saved) : block)
                    .toList();
            return message.withContent(blocks);
        }).toList();
    }

    private ToolResultBlock project(ToolResultBlock result, RuntimeContext context, Map<String, String> saved) {
        if (ReadToolResultToolExecutor.NAME.equals(result.getName())
                || ReadScriptContentToolExecutor.NAME.equals(result.getName())) {
            return result;
        }
        StringBuilder text = new StringBuilder();
        for (ContentBlock block : result.getOutput()) {
            if (block instanceof TextBlock part) {
                text.append(part.getText());
            }
        }
        String fullText = text.toString();
        if (fullText.codePointCount(0, fullText.length()) <= maxInlineChars) {
            return result;
        }
        String hash = TextPages.version(fullText);
        String reference = saved.computeIfAbsent(hash, ignored -> {
            String id = results.save(context.getUserId(), context.getSessionId(), fullText);
            try {
                return json.writeValueAsString(Map.of(
                        "resultId", id, "contentVersion", id,
                        "totalChars", fullText.codePointCount(0, fullText.length()),
                        "readTool", ReadToolResultToolExecutor.NAME, "offset", 0,
                        "message", "完整结果已保存。请分段读取；尚未读取的部分不能作为处理或完成的依据。"));
            } catch (JsonProcessingException failure) {
                throw new IllegalStateException("无法生成长工具结果的读取入口", failure);
            }
        });
        List<ContentBlock> output = new ArrayList<>();
        boolean replaced = false;
        for (ContentBlock block : result.getOutput()) {
            if (block instanceof TextBlock) {
                if (!replaced) {
                    output.add(TextBlock.builder().text(reference).build());
                    replaced = true;
                }
            } else {
                output.add(block);
            }
        }
        return new ToolResultBlock(result.getId(), result.getName(), output, result.getMetadata(), result.getState());
    }
}
