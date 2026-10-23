package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.canvas.CanvasTextJobEntity;
import com.stonewu.fusion.mapper.canvas.CanvasTextJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasGenerationUsageMapper;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

/**
 * 从同一读取事务内的执行事实统计，不将预算、复用或转发结果计为新增产物。
 */
@Component
@RequiredArgsConstructor
public class CanvasRunStatistics {
    private final CanvasTextJobMapper texts;
    private final CanvasGenerationUsageMapper generations;
    private final CanvasJson json;

    public record Nodes(int total, int succeeded, int failed, int blocked, int cancelled,
                        int pending, int reused, int batchAttempts, int batchRetries) {
    }

    public record Media(int images, int videos, int audios, long videoDurationMs, long audioDurationMs,
                        int unknownDurations, long bytes, int unavailableNodes) {
    }

    public record Tokens(int submittedCalls, int measuredCalls, Long input, Long output, Long total) {
    }

    public record Statistics(Instant observedAt, long elapsedMs, Nodes nodes, Media media, Tokens tokens) {
    }

    public Statistics read(Run run, List<NodeRun> nodes, Set<String> unavailable) {
        var usage = new ArrayList<JsonNode>();
        texts.selectList(new LambdaQueryWrapper<CanvasTextJobEntity>()
                .select(CanvasTextJobEntity::getId, CanvasTextJobEntity::getResultJson)
                .eq(CanvasTextJobEntity::getRunId, run.id()).isNotNull(CanvasTextJobEntity::getRequestStartedAt)
                .orderByAsc(CanvasTextJobEntity::getId)).forEach(row -> {
            String result = row.getResultJson();
            usage.add(result == null ? json.object(Map.of()) : json.read(result, JsonNode.class).path("usage"));
        });
        for (var row : generations.tokensForRun(run.id())) {
            Long input = row.inputTokens(), output = row.outputTokens();
            var item = json.object(Map.of());
            if (input != null && output != null) {
                item.put("inputTokens", input).put("outputTokens", output).put("totalTokens", input + output);
            }
            usage.add(item);
        }
        return measure(run, nodes, usage, Instant.now(), unavailable);
    }

    static Statistics measure(Run run, List<NodeRun> nodes, List<JsonNode> usage, Instant now,
                              Set<String> unavailable) {
        int total = 0, success = 0, failed = 0, blocked = 0, cancelled = 0, pending = 0, reused = 0;
        int attempts = 0, retries = 0;
        Map<String, Value> produced = new HashMap<>();
        for (var node : nodes) {
            if (node.parentNodeRunId() == null) {
                total++;
                switch (node.status()) {
                    case SUCCEEDED -> success++;
                    case FAILED, PARTIAL_FAILED -> failed++;
                    case BLOCKED -> blocked++;
                    case CANCELLED -> cancelled++;
                    default -> pending++;
                }
                if (node.plan().action().equals("REUSE") && node.status() == State.SUCCEEDED) {
                    reused++;
                }
            } else {
                attempts++;
                if (node.attempt() > 1) {
                    retries++;
                }
            }
            collectProduced(node, unavailable, produced);
        }
        Media media = media(produced, unavailable.size());
        int measured = 0;
        long input = 0, output = 0, tokens = 0;
        for (var item : usage) {
            if (!known(item.path("inputTokens")) || !known(item.path("outputTokens")) || !known(
                    item.path("totalTokens"))) {
                continue;
            }
            measured++;
            input += item.path("inputTokens").longValue();
            output += item.path("outputTokens").longValue();
            tokens += item.path("totalTokens").longValue();
        }
        Instant end = run.status().terminal() ? run.updatedAt() : now;
        return new Statistics(now, Math.max(0, Duration.between(run.createdAt(), end).toMillis()),
                new Nodes(total, success, failed, blocked, cancelled, pending, reused, attempts, retries), media,
                new Tokens(usage.size(), measured, measured == 0 ? null : input, measured == 0 ? null : output,
                        measured == 0 ? null : tokens));
    }

    static void collectProduced(NodeRun node, Set<String> unavailable, Map<String, Value> produced) {
        if (!unavailable.contains(node.id()) && node.plan().action().equals("EXECUTE") && node.externalId() != null
                && Set.of("GENERATION", "MEDIA", "AGENT").contains(node.externalKind())) {
            for (var value : node.outputs().all()) {
                if (value.mediaId() != null) {
                    produced.putIfAbsent(value.mediaId() + ":" + value.mediaVersion(), value);
                }
            }
        }
    }

    static Media media(Map<String, Value> produced, int unavailable) {
        int images = 0, videos = 0, audios = 0, unknown = 0;
        long videoMs = 0, audioMs = 0, bytes = 0;
        for (var value : produced.values()) {
            var content = value.data().path("content");
            bytes += content.path("byteSize").longValue();
            if (value.type() == CanvasValueType.IMAGE) {
                images++;
            } else if (value.type() == CanvasValueType.VIDEO) {
                videos++;
                if (content.path("durationMs").isIntegralNumber()) {
                    videoMs += content.path("durationMs").longValue();
                } else {
                    unknown++;
                }
            } else if (value.type() == CanvasValueType.AUDIO) {
                audios++;
                if (content.path("durationMs").isIntegralNumber()) {
                    audioMs += content.path("durationMs").longValue();
                } else {
                    unknown++;
                }
            }
        }
        return new Media(images, videos, audios, videoMs, audioMs, unknown, bytes, unavailable);
    }

    private static boolean known(JsonNode value) {
        return value.isIntegralNumber() && value.canConvertToLong() && value.longValue() >= 0;
    }
}
