package com.stonewu.fusion.service.ai.provider;

import com.anthropic.models.messages.RawMessageStreamEvent;
import com.anthropic.models.messages.StopReason;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ThinkingBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ChatUsage;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 保留 Anthropic 的完成原因、累计用量和工具 index，空内容的用量事件也必须交给调用登记。
 */
final class StreamingAnthropicResponses {
    private final Instant started;
    private final Map<Long, ToolIdentity> tools = new HashMap<>();
    private String messageId;
    private int inputTokens;
    private int cachedTokens;

    StreamingAnthropicResponses(Instant started) {
        this.started = started;
    }

    List<ChatResponse> accept(RawMessageStreamEvent event) {
        if (event.isMessageStart()) {
            var message = event.asMessageStart().message();
            messageId = message.id();
            inputTokens = Math.toIntExact(message.usage().inputTokens());
            cachedTokens = Math.toIntExact(message.usage().cacheReadInputTokens().orElse(0L));
            return List.of();
        }
        List<ContentBlock> blocks = new ArrayList<>();
        if (event.isContentBlockStart()) {
            var start = event.asContentBlockStart();
            start.contentBlock().toolUse().ifPresent(tool -> {
                tools.put(start.index(), new ToolIdentity(tool.id(), tool.name()));
                blocks.add(ToolUseBlock.builder().id(tool.id()).name(tool.name()).input(Map.of()).content("").build());
            });
        }
        if (event.isContentBlockDelta()) {
            var delta = event.asContentBlockDelta();
            delta.delta().text().ifPresent(text -> blocks.add(TextBlock.builder().text(text.text()).build()));
            delta.delta().thinking()
                    .ifPresent(thinking -> blocks.add(ThinkingBlock.builder().thinking(thinking.thinking()).build()));
            delta.delta().inputJson().ifPresent(arguments -> {
                ToolIdentity tool = tools.get(delta.index());
                if (tool == null) {
                    throw new IllegalStateException("Anthropic 工具参数片段缺少所属内容块");
                }
                blocks.add(ToolUseBlock.builder().id(tool.id()).name(tool.name()).input(Map.of())
                        .content(arguments.partialJson()).build());
            });
        }
        ChatUsage usage = null;
        String finish = null;
        if (event.isMessageDelta()) {
            var delta = event.asMessageDelta();
            usage = ChatUsage.builder().inputTokens(inputTokens).cachedTokens(cachedTokens)
                    .outputTokens(Math.toIntExact(delta.usage().outputTokens()))
                    .time(Duration.between(started, Instant.now()).toMillis() / 1000.0).build();
            finish = delta.delta().stopReason().map(StopReason::asString).orElse(null);
        }
        if (blocks.isEmpty() && usage == null && finish == null) {
            return List.of();
        }
        if (messageId == null) {
            throw new IllegalStateException("Anthropic 内容先于消息开始事件到达");
        }
        return List.of(ChatResponse.builder().id(messageId).content(blocks).usage(usage).finishReason(finish).build());
    }

    private record ToolIdentity(String id, String name) {
    }
}
