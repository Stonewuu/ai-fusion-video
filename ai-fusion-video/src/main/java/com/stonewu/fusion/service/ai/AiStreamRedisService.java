package com.stonewu.fusion.service.ai;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.connection.stream.ReadOffset;
import org.springframework.data.redis.connection.stream.RecordId;
import org.springframework.data.redis.connection.stream.StreamOffset;
import org.springframework.data.redis.connection.stream.StreamReadOptions;
import org.springframework.data.redis.connection.stream.StreamRecords;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * AI 流式对话 Redis Stream 服务
 * <p>
 * 双通道架构：
 * - Redis Stream：实时逐 token 传输（初始连接使用）
 * - Redis List（Replay）：合并后的逻辑事件（重连使用）
 * <p>
 * 重连时先从 Replay List 回放完整历史（瞬间），
 * 然后从 Stream 的 lastStreamId 位置续传实时 token。
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AiStreamRedisService {

    private final StringRedisTemplate stringRedisTemplate;

    private static final String STREAM_KEY_PREFIX = "fv:ai:stream:";
    private static final String STATUS_KEY_PREFIX = "fv:ai:stream:status:";
    private static final String REPLAY_KEY_PREFIX = "fv:ai:stream:replay:";

    /** 状态过期时间：1小时 */
    private static final Duration STATUS_TTL = Duration.ofHours(1);
    /** Stream / Replay List 过期时间：1小时 */
    private static final Duration STREAM_TTL = Duration.ofHours(1);
    /** Stream 最大长度 —— 只需保存最近一次合并到当前的增量 token */
    private static final long STREAM_MAXLEN = 200;
    /** XREAD BLOCK 超时时间。调低以减少实时 SSE 的感知延迟。 */
    private static final Duration READ_BLOCK_TIMEOUT = Duration.ofMillis(50);
    /** 每次 XREAD 读取的最大消息数。保持为 1 以尽量按单事件透传。 */
    private static final int READ_BATCH_SIZE = 1;

    public static boolean isMainAgentTerminalEvent(AiChatStreamRespVO event) {
        return event != null
                && StrUtil.isBlank(event.getParentToolCallId())
                && StrUtil.isBlank(event.getAgentName())
                && ("DONE".equals(event.getOutputType())
                        || "ERROR".equals(event.getOutputType())
                        || "CANCELLED".equals(event.getOutputType()));
    }

    public static boolean isMainAgentErrorEvent(AiChatStreamRespVO event) {
        return event != null
                && StrUtil.isBlank(event.getParentToolCallId())
                && StrUtil.isBlank(event.getAgentName())
                && "ERROR".equals(event.getOutputType());
    }

    public static boolean isMainAgentCancelledEvent(AiChatStreamRespVO event) {
        return event != null
                && StrUtil.isBlank(event.getParentToolCallId())
                && StrUtil.isBlank(event.getAgentName())
                && "CANCELLED".equals(event.getOutputType());
    }

    // ===== 写入端（后台 AI 任务调用） =====

    /**
     * 发布事件到 Redis Stream（实时通道）
     *
     * @return 写入的 Stream 记录 ID，供 Accumulator 记录 lastStreamId
     */
    public String publish(String conversationId, AiChatStreamRespVO event) {
        String streamKey = STREAM_KEY_PREFIX + conversationId;
        Map<String, String> eventMap = Map.of("data", JSONUtil.toJsonStr(event));
        try {
            RecordId recordId = stringRedisTemplate.opsForStream().add(
                    StreamRecords.string(eventMap).withStreamKey(streamKey));
            // 精确裁剪保持长度可控
            stringRedisTemplate.opsForStream().trim(streamKey, STREAM_MAXLEN, false);
            return recordId != null ? recordId.getValue() : null;
        } catch (Exception e) {
            log.error("发布事件到 Redis Stream 失败: conversationId={}", conversationId, e);
            return null;
        }
    }

    // ===== 读取端（SSE 连接调用） =====

    /**
     * 从 Stream 起始位置订阅所有事件（首次连接用）
     */
    public Flux<AiChatStreamRespVO> subscribe(String conversationId) {
        return subscribeFrom(conversationId, "0-0");
    }

    /**
     * 从指定位置订阅事件
     * <p>
     * 使用虚拟线程阻塞读取 Redis Stream，通过 Flux.create 桥接为响应式流。
     */
    @SuppressWarnings("unchecked")
    public Flux<AiChatStreamRespVO> subscribeFrom(String conversationId, String lastEventId) {
        String streamKey = STREAM_KEY_PREFIX + conversationId;
        return Flux.<AiChatStreamRespVO>create(sink -> {
            Thread.startVirtualThread(() -> {
                String currentId = lastEventId;
                log.info("开始从 Redis Stream 读取事件: conversationId={}, startId={}",
                        conversationId, lastEventId);
                while (!sink.isCancelled()) {
                    try {
                        List<MapRecord<String, String, String>> records =
                                (List<MapRecord<String, String, String>>) (List<?>) stringRedisTemplate
                                        .opsForStream()
                                        .read(StreamReadOptions.empty()
                                                        .count(READ_BATCH_SIZE)
                                                        .block(READ_BLOCK_TIMEOUT),
                                                StreamOffset.create(streamKey,
                                                        ReadOffset.from(currentId)));

                        if (records != null && !records.isEmpty()) {
                            for (MapRecord<String, String, String> record : records) {
                                currentId = record.getId().getValue();
                                String json = record.getValue().get("data");
                                if (json != null) {
                                    AiChatStreamRespVO event = JSONUtil.toBean(json,
                                            AiChatStreamRespVO.class);
                                    sink.next(event);
                                    // 仅主 Agent 的终态事件才结束整个会话流。
                                    if (isMainAgentTerminalEvent(event)) {
                                        log.info("读取到终止事件: conversationId={}, type={}",
                                                conversationId, event.getOutputType());
                                        sink.complete();
                                        return;
                                    }
                                }
                            }
                        } else {
                            // 没有新消息，检查会话是否已结束
                            if (!isActive(conversationId)) {
                                log.info("会话已非活跃状态，结束 Stream 读取: conversationId={}",
                                        conversationId);
                                sink.complete();
                                return;
                            }
                        }
                    } catch (Exception e) {
                        if (!sink.isCancelled()) {
                            log.error("从 Redis Stream 读取事件失败: conversationId={}",
                                    conversationId, e);
                            sink.error(e);
                        }
                        return;
                    }
                }
                log.info("SSE 订阅被取消，退出 Stream 读取: conversationId={}", conversationId);
            });
        }).doFinally(signal -> log.debug("Stream 订阅结束: conversationId={}, signal={}",
                conversationId, signal));
    }

    // ===== Replay List（合并后的逻辑事件） =====

    /**
     * 追加合并事件到 Replay List
     *
     * @param lastStreamId 该合并事件对应的最后一条 Stream 记录 ID
     */
    public void appendReplayEvent(String conversationId, AiChatStreamRespVO event,
            String lastStreamId) {
        String replayKey = REPLAY_KEY_PREFIX + conversationId;
        try {
            JSONObject wrapper = new JSONObject();
            wrapper.set("event", JSONUtil.toJsonStr(event));
            wrapper.set("lastStreamId", lastStreamId);
            stringRedisTemplate.opsForList().rightPush(replayKey, wrapper.toString());
        } catch (Exception e) {
            log.error("追加 Replay 事件失败: conversationId={}", conversationId, e);
        }
    }

    /**
     * 读取所有合并事件（重连回放用）
     *
     * @return (events, lastStreamId) 元组；如果没有事件，返回空列表和 "0-0"
     */
    public ReplayResult getReplayEvents(String conversationId) {
        String replayKey = REPLAY_KEY_PREFIX + conversationId;
        try {
            List<String> raw = stringRedisTemplate.opsForList().range(replayKey, 0, -1);
            if (raw == null || raw.isEmpty()) {
                return new ReplayResult(Collections.emptyList(), "0-0");
            }

            List<AiChatStreamRespVO> events = new ArrayList<>(raw.size());
            String lastStreamId = "0-0";
            for (String json : raw) {
                JSONObject wrapper = JSONUtil.parseObj(json);
                String eventJson = wrapper.getStr("event");
                String streamId = wrapper.getStr("lastStreamId");
                if (eventJson != null) {
                    events.add(JSONUtil.toBean(eventJson, AiChatStreamRespVO.class));
                }
                if (StrUtil.isNotBlank(streamId)) {
                    lastStreamId = streamId;
                }
            }
            return new ReplayResult(events, lastStreamId);
        } catch (Exception e) {
            log.error("读取 Replay 事件失败: conversationId={}", conversationId, e);
            return new ReplayResult(Collections.emptyList(), "0-0");
        }
    }

    /** Replay 读取结果 */
    @Getter
    @RequiredArgsConstructor
    public static class ReplayResult {
        private final List<AiChatStreamRespVO> events;
        private final String lastStreamId;
    }

    // ===== 状态管理 =====

    /**
     * 清理会话的所有 Redis 数据（续聊时调用）
     */
    public void cleanup(String conversationId) {
        String streamKey = STREAM_KEY_PREFIX + conversationId;
        String statusKey = STATUS_KEY_PREFIX + conversationId;
        String replayKey = REPLAY_KEY_PREFIX + conversationId;
        try {
            stringRedisTemplate.delete(streamKey);
            stringRedisTemplate.delete(statusKey);
            stringRedisTemplate.delete(replayKey);
            log.debug("已清理旧 Stream/Replay 数据: conversationId={}", conversationId);
        } catch (Exception e) {
            log.warn("清理旧 Stream 数据失败: conversationId={}", conversationId, e);
        }
    }

    /** 标记会话为活跃 */
    public void markActive(String conversationId) {
        stringRedisTemplate.opsForValue()
                .set(STATUS_KEY_PREFIX + conversationId, "ACTIVE", STATUS_TTL);
        log.info("标记会话为 ACTIVE: conversationId={}", conversationId);
    }

    /** 标记会话为已完成 */
    public void markCompleted(String conversationId) {
        stringRedisTemplate.opsForValue()
                .set(STATUS_KEY_PREFIX + conversationId, "COMPLETED", STATUS_TTL);
        log.info("标记会话为 COMPLETED: conversationId={}", conversationId);
    }

    /** 标记会话为错误 */
    public void markError(String conversationId) {
        stringRedisTemplate.opsForValue()
                .set(STATUS_KEY_PREFIX + conversationId, "ERROR", STATUS_TTL);
        log.info("标记会话为 ERROR: conversationId={}", conversationId);
    }

    /** 检查会话是否活跃 */
    public boolean isActive(String conversationId) {
        return "ACTIVE".equals(
                stringRedisTemplate.opsForValue().get(STATUS_KEY_PREFIX + conversationId));
    }

    /**
     * 获取会话流式状态
     *
     * @return ACTIVE / COMPLETED / ERROR / NONE
     */
    public String getStatus(String conversationId) {
        String status = stringRedisTemplate.opsForValue()
                .get(STATUS_KEY_PREFIX + conversationId);
        return status != null ? status : "NONE";
    }

    /** 安排清理：给 Stream、Replay List 和状态 Key 设置 TTL */
    public void scheduleCleanup(String conversationId) {
        String streamKey = STREAM_KEY_PREFIX + conversationId;
        String replayKey = REPLAY_KEY_PREFIX + conversationId;
        try {
            stringRedisTemplate.expire(streamKey, STREAM_TTL);
            stringRedisTemplate.expire(replayKey, STREAM_TTL);
            log.debug("已为 Stream/Replay 设置 TTL: conversationId={}", conversationId);
        } catch (Exception e) {
            log.warn("设置 Stream TTL 失败: conversationId={}", conversationId, e);
        }
    }

    // ===== 事件合并累积器 =====

    /**
     * 将连续的 REASONING/CONTENT token 事件合并为单条完整文本记录。
     * <p>
     * 合并规则：连续且 (outputType, parentToolCallId) 相同的 REASONING/CONTENT 事件
     * 会被累积。当类型切换或遇到离散事件（TOOL_CALL/TOOL_FINISHED/DONE/ERROR）时，
     * 先 flush 已积累的文本为一条合并记录，然后透传离散事件。
     * <p>
     * 使用方法：
     * <pre>
     * var acc = new StreamEventAccumulator(conversationId);
     * // 对每个事件
     * List<AccumulatedEvent> flushed = acc.accumulate(event, streamId);
     * for (AccumulatedEvent ae : flushed) {
     *     aiStreamRedisService.appendReplayEvent(conversationId, ae.event, ae.lastStreamId);
     * }
     * // 结束时
     * AccumulatedEvent remaining = acc.flush();
     * if (remaining != null) { ... }
     * </pre>
     */
    public static class StreamEventAccumulator {

        private final String conversationId;

        // 当前正在累积的事件模板（outputType/parentToolCallId/messageId/conversationId/agentName 等）
        private AiChatStreamRespVO currentTemplate;
        private String currentOutputType;
        private String currentParentToolCallId;
        private final StringBuilder textBuffer = new StringBuilder();
        private String lastStreamId;
        // 记录第一个 REASONING 事件的 startTime
        private Long reasoningStartTime;
        // 记录最新的 reasoningDurationMs
        private Long reasoningDurationMs;

        public StreamEventAccumulator(String conversationId) {
            this.conversationId = conversationId;
        }

        /**
         * 累积一个事件。
         *
         * @param event    原始 SSE 事件
         * @param streamId 该事件在 Redis Stream 中的 ID
         * @return 需要写入 Replay List 的合并事件列表（可能为空或包含 1-2 个事件）
         */
        public List<AccumulatedEvent> accumulate(AiChatStreamRespVO event, String streamId) {
            String outputType = event.getOutputType();
            String parentToolCallId = event.getParentToolCallId();
            List<AccumulatedEvent> result = new ArrayList<>(2);

            if (isMergeableType(outputType)) {
                // REASONING 或 CONTENT：检查是否与当前累积的类型/上下文相同
                boolean sameContext = outputType.equals(currentOutputType)
                        && StrUtil.equals(parentToolCallId, currentParentToolCallId);

                if (sameContext) {
                    // 继续累积
                    appendText(event);
                    this.lastStreamId = streamId;

                    // 追踪 reasoning 耗时
                    if ("REASONING".equals(outputType)) {
                        if (event.getReasoningStartTime() != null && reasoningStartTime == null) {
                            reasoningStartTime = event.getReasoningStartTime();
                        }
                    }
                    if ("CONTENT".equals(outputType) && event.getReasoningDurationMs() != null) {
                        reasoningDurationMs = event.getReasoningDurationMs();
                    }
                } else {
                    // 类型/上下文变了 → 先 flush 旧的
                    AccumulatedEvent flushed = flush();
                    if (flushed != null) {
                        result.add(flushed);
                    }
                    // 开始新的累积
                    startNewAccumulation(event, outputType, parentToolCallId, streamId);
                }
            } else {
                // 离散事件（TOOL_CALL, TOOL_FINISHED, DONE, ERROR, CANCELLED 等）
                // 先 flush 累积的内容
                AccumulatedEvent flushed = flush();
                if (flushed != null) {
                    result.add(flushed);
                }
                // 离散事件直接透传
                result.add(new AccumulatedEvent(event, streamId));
            }

            return result;
        }

        /**
         * 刷新当前累积的内容为一条合并事件
         *
         * @return 合并事件，如果无积累则返回 null
         */
        public AccumulatedEvent flush() {
            if (currentTemplate == null || textBuffer.isEmpty()) {
                // 清空状态
                reset();
                return null;
            }

            // 构建合并后的事件
            AiChatStreamRespVO merged = new AiChatStreamRespVO()
                    .setMessageId(currentTemplate.getMessageId())
                    .setConversationId(conversationId)
                    .setOutputType(currentOutputType)
                    .setParentToolCallId(currentParentToolCallId)
                    .setAgentName(currentTemplate.getAgentName());

            if ("REASONING".equals(currentOutputType)) {
                merged.setReasoningContent(textBuffer.toString());
                if (reasoningStartTime != null) {
                    merged.setReasoningStartTime(reasoningStartTime);
                }
            } else if ("CONTENT".equals(currentOutputType)) {
                merged.setContent(textBuffer.toString());
                if (reasoningDurationMs != null) {
                    merged.setReasoningDurationMs(reasoningDurationMs);
                }
            }

            String sid = this.lastStreamId;
            reset();
            return new AccumulatedEvent(merged, sid);
        }

        private void startNewAccumulation(AiChatStreamRespVO event, String outputType,
                String parentToolCallId, String streamId) {
            this.currentTemplate = event;
            this.currentOutputType = outputType;
            this.currentParentToolCallId = parentToolCallId;
            this.lastStreamId = streamId;
            this.reasoningStartTime = null;
            this.reasoningDurationMs = null;
            textBuffer.setLength(0);
            appendText(event);

            if ("REASONING".equals(outputType) && event.getReasoningStartTime() != null) {
                reasoningStartTime = event.getReasoningStartTime();
            }
            if ("CONTENT".equals(outputType) && event.getReasoningDurationMs() != null) {
                reasoningDurationMs = event.getReasoningDurationMs();
            }
        }

        private void appendText(AiChatStreamRespVO event) {
            if ("REASONING".equals(currentOutputType)) {
                if (StrUtil.isNotEmpty(event.getReasoningContent())) {
                    textBuffer.append(event.getReasoningContent());
                }
            } else if ("CONTENT".equals(currentOutputType)) {
                if (event.getContent() != null) {
                    textBuffer.append(event.getContent());
                }
            }
        }

        private void reset() {
            currentTemplate = null;
            currentOutputType = null;
            currentParentToolCallId = null;
            lastStreamId = null;
            reasoningStartTime = null;
            reasoningDurationMs = null;
            textBuffer.setLength(0);
        }

        private boolean isMergeableType(String outputType) {
            return "REASONING".equals(outputType) || "CONTENT".equals(outputType);
        }

        /** 合并后的事件 + 对应的 lastStreamId */
        @Getter
        @RequiredArgsConstructor
        public static class AccumulatedEvent {
            private final AiChatStreamRespVO event;
            private final String lastStreamId;
        }
    }
}
