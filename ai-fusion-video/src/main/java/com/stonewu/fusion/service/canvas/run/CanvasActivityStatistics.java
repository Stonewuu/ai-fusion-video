package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportRepository;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportSources;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.entity.canvas.CanvasRunEntity;
import com.stonewu.fusion.entity.canvas.CanvasExportEntity;
import com.stonewu.fusion.mapper.canvas.CanvasRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasExportMapper;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.CanvasRunStatistics.*;

/**
 * 时间范围内完整汇总，不默默截断历史；权限边界与单次运行/交付保持一致。
 */
@Component
@RequiredArgsConstructor
public class CanvasActivityStatistics {
    private final CanvasRunMapper runRows;
    private final CanvasExportMapper exportRows;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasRunRepository runs;
    private final CanvasRunSources sources;
    private final CanvasRunStatistics statistics;
    private final CanvasExportRepository exports;
    private final CanvasExportSources exportSources;

    public record Delivery(int total, Map<String, Integer> states, long bytes, int unavailable) {
    }

    public record Activity(Instant from, Instant until, Instant observedAt, int runs, Map<String, Integer> states,
                           long elapsedMs, Nodes nodes, Media media, Tokens tokens, Delivery deliveries) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Activity read(String canvasId, Instant from, Instant until, CanvasActor actor) {
        access.requireView(documents.require(canvasId, false), actor);
        if (from == null || until == null || !until.isAfter(from) || Duration.between(from, until)
                .compareTo(Duration.ofDays(366)) > 0) {
            throw CanvasRunException.invalid("请选择不超过一年的有效统计时间范围");
        }
        var start = LocalDateTime.ofInstant(from, ZoneOffset.UTC);
        var end = LocalDateTime.ofInstant(until, ZoneOffset.UTC);
        var ids = runRows.selectPage(new Page<CanvasRunEntity>(1, 1001, false),
                        new LambdaQueryWrapper<CanvasRunEntity>()
                                .select(CanvasRunEntity::getId).eq(CanvasRunEntity::getCanvasId, canvasId)
                                .ge(CanvasRunEntity::getCreatedAt, start).lt(CanvasRunEntity::getCreatedAt, end)
                                .orderByAsc(CanvasRunEntity::getCreatedAt, CanvasRunEntity::getId))
                .getRecords().stream().map(CanvasRunEntity::getId).toList();
        var deliveryIds = exportRows.selectPage(new Page<CanvasExportEntity>(1, 1001, false),
                        new LambdaQueryWrapper<CanvasExportEntity>()
                                .select(CanvasExportEntity::getId).eq(CanvasExportEntity::getCanvasId, canvasId)
                                .ge(CanvasExportEntity::getCreateTime, start).lt(CanvasExportEntity::getCreateTime, end)
                                .orderByAsc(CanvasExportEntity::getCreateTime, CanvasExportEntity::getId))
                .getRecords().stream().map(CanvasExportEntity::getId).toList();
        if (ids.size() > 1000 || deliveryIds.size() > 1000) {
            throw CanvasRunException.invalid("此范围超过 1000 次运行或交付，请缩短时间范围；统计不会截断记录");
        }
        Instant observed = Instant.now();
        var accumulator = new Accumulator();
        for (String id : ids) {
            var run = runs.run(id, false);
            var nodes = runs.nodes(id);
            var unavailable = new HashSet<String>();
            nodes.forEach(node -> {
                if (!sources.visible(node, actor)) {
                    unavailable.add(node.id());
                }
            });
            accumulator.add(run.status().name(), statistics.read(run, nodes, unavailable));
            nodes.forEach(node -> collectProduced(node, unavailable, accumulator.produced));
        }
        var deliveryStates = new HashMap<String, Integer>();
        long bytes = 0;
        int unavailable = 0;
        for (String id : deliveryIds) {
            var job = exports.get(id, false);
            deliveryStates.merge(job.status(), 1, Integer::sum);
            try {
                exportSources.authorize(canvasId, job.manifest(), actor);
                if (job.status().equals("SUCCEEDED") && job.byteSize() != null) {
                    bytes += job.byteSize();
                }
            } catch (BusinessException failure) {
                if (!List.of(403, 404, 409, 410).contains(failure.getCode())) {
                    throw failure;
                }
                unavailable++;
            }
        }
        return new Activity(from, until, observed, ids.size(), Map.copyOf(accumulator.states), accumulator.elapsed,
                accumulator.nodes, media(accumulator.produced, accumulator.unavailable), accumulator.tokens(),
                new Delivery(deliveryIds.size(), Map.copyOf(deliveryStates), bytes, unavailable));
    }

    static final class Accumulator {
        final Map<String, Integer> states = new HashMap<>();
        final Map<String, Value> produced = new HashMap<>();
        Nodes nodes = new Nodes(0, 0, 0, 0, 0, 0, 0, 0, 0);
        long elapsed, input, output, totalTokens;
        int submitted, measured, unavailable;

        void add(String status, Statistics stats) {
            states.merge(status, 1, Integer::sum);
            elapsed += stats.elapsedMs();
            var next = stats.nodes();
            nodes = new Nodes(nodes.total() + next.total(), nodes.succeeded() + next.succeeded(),
                    nodes.failed() + next.failed(),
                    nodes.blocked() + next.blocked(), nodes.cancelled() + next.cancelled(),
                    nodes.pending() + next.pending(),
                    nodes.reused() + next.reused(), nodes.batchAttempts() + next.batchAttempts(),
                    nodes.batchRetries() + next.batchRetries());
            var tokens = stats.tokens();
            submitted += tokens.submittedCalls();
            measured += tokens.measuredCalls();
            if (tokens.input() != null) {
                input += tokens.input();
            }
            if (tokens.output() != null) {
                output += tokens.output();
            }
            if (tokens.total() != null) {
                totalTokens += tokens.total();
            }
            unavailable += stats.media().unavailableNodes();
        }

        Tokens tokens() {
            return new Tokens(submitted, measured, measured == 0 ? null : input, measured == 0 ? null : output,
                    measured == 0 ? null : totalTokens);
        }
    }
}
