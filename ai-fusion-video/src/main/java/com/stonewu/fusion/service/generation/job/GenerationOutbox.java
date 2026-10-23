package com.stonewu.fusion.service.generation.job;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.stonewu.fusion.mapper.generation.AsyncOutboxMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasTextJobMapper;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import com.stonewu.fusion.entity.media.MediaJobEntity;
import com.stonewu.fusion.entity.canvas.CanvasRunEntity;
import com.stonewu.fusion.entity.canvas.CanvasTextJobEntity;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.Map;

@Component
@RequiredArgsConstructor
@Slf4j
public class GenerationOutbox {
    private final AsyncOutboxMapper entries;
    private final ExecutionCoordinationMapper coordination;
    private final RemoteGenerationJobMapper remoteJobs;
    private final MediaJobMapper mediaJobs;
    private final CanvasRunMapper canvasRuns;
    private final CanvasTextJobMapper textJobs;
    private final CanvasJson json;
    private final TransactionTemplate transactions;

    @Transactional(propagation = Propagation.MANDATORY)
    public void enqueue(String key, String type, String aggregate, Object payload) {
        entries.enqueue(key, type, aggregate, json.write(payload));
    }

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        for (int i = 0; i < 32; i++) {
            try {
                Boolean published = transactions.execute(status -> publishOne());
                if (!Boolean.TRUE.equals(published)) {
                    return;
                }
            } catch (DeliveryFailure failure) {
                transactions.executeWithoutResult(ignored -> {
                    coordination.lockGenerationDispatch();
                    entries.backoff(failure.id, "事件投递失败：" + failure.getCause().getClass().getSimpleName());
                });
                log.warn("任务事件投递暂未完成，将按退避时间重试 [outboxId={}]", failure.id, failure.getCause());
            }
        }
    }

    /**
     * 同一锁直到事务提交才释放，续读游标不会越过尚未提交的更小事件。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public boolean publishOne() {
        coordination.lockGenerationDispatch();
        var row = entries.nextPending();
        if (row == null) {
            return false;
        }
        var entry = new Entry(row.getId(), row.getEventType(), row.getAggregateId(),
                json.read(row.getPayloadJson(), ObjectNode.class));
        try {
            deliver(entry);
        } catch (RuntimeException failure) {
            throw new DeliveryFailure(entry.id(), failure);
        }
        return true;
    }

    private void deliver(Entry entry) {
        if (entry.type().equals("GENERATION_ACCEPTED")) {
            remoteJobs.update(new LambdaUpdateWrapper<RemoteGenerationJobEntity>()
                    .eq(RemoteGenerationJobEntity::getSubmissionId, entry.aggregate())
                    .eq(RemoteGenerationJobEntity::getState, "QUEUED")
                    .set(RemoteGenerationJobEntity::getState, "READY"));
        } else if (entry.type().equals("COMFY_TEST_ACCEPTED")) {
            mediaJobs.update(new LambdaUpdateWrapper<MediaJobEntity>()
                    .eq(MediaJobEntity::getId, entry.aggregate()).eq(MediaJobEntity::getStatus, "QUEUED")
                    .eq(MediaJobEntity::getKind, "COMFY_WORKFLOW_TEST").set(MediaJobEntity::getStatus, "WAITING"));
        } else if (entry.type().equals("CANVAS_RUN_ACCEPTED")) {
            canvasRuns.update(new LambdaUpdateWrapper<CanvasRunEntity>()
                    .eq(CanvasRunEntity::getId, entry.aggregate()).eq(CanvasRunEntity::getStatus, "QUEUED")
                    .set(CanvasRunEntity::getStatus, "RUNNING"));
        } else if (entry.type().equals("CANVAS_MEDIA_ACCEPTED")) {
            mediaJobs.update(new LambdaUpdateWrapper<MediaJobEntity>()
                    .eq(MediaJobEntity::getId, entry.aggregate()).eq(MediaJobEntity::getStatus, "QUEUED")
                    .eq(MediaJobEntity::getKind, "CANVAS_MEDIA").set(MediaJobEntity::getStatus, "WAITING"));
        } else if (entry.type().equals("CANVAS_TEXT_ACCEPTED")) {
            textJobs.update(new LambdaUpdateWrapper<CanvasTextJobEntity>()
                    .eq(CanvasTextJobEntity::getId, entry.aggregate()).eq(CanvasTextJobEntity::getStatus, "QUEUED")
                    .set(CanvasTextJobEntity::getStatus, "READY"));
        } else if (entry.type().equals("MEDIA_IMPORT_ACCEPTED") || entry.type().equals("MEDIA_IMPORT_RETRY")) {
            mediaJobs.update(new LambdaUpdateWrapper<MediaJobEntity>()
                    .eq(MediaJobEntity::getId, entry.aggregate()).eq(MediaJobEntity::getStatus, "QUEUED")
                    .eq(MediaJobEntity::getKind, "SOURCE_IMPORT").set(MediaJobEntity::getStatus, "WAITING"));
        } else if (entry.type().equals("MEDIA_PREVIEW_ACCEPTED")) {
            mediaJobs.update(new LambdaUpdateWrapper<MediaJobEntity>()
                    .eq(MediaJobEntity::getId, entry.aggregate()).eq(MediaJobEntity::getStatus, "QUEUED")
                    .eq(MediaJobEntity::getKind, "MEDIA_PREVIEW").set(MediaJobEntity::getStatus, "WAITING"));
        } else if (!entry.type().equals("GENERATION_CHANGED") && !entry.type().startsWith("COMFY_TEST_")
                && !entry.type().startsWith("CANVAS_RUN_") && !entry.type().startsWith("CANVAS_MEDIA_") && !entry.type()
                .startsWith("CANVAS_TEXT_")
                && !entry.type().equals("EPISODE_AUDIO_REVIEW") && !entry.type()
                .startsWith("MEDIA_IMPORT_") && !entry.type().startsWith("MEDIA_PREVIEW_")) {
            throw new IllegalStateException("发件箱出现未登记的事件类型: " + entry.type());
        }
        String eventStream = entry.type().startsWith("CANVAS_TEXT_") ? "CANVAS_TEXT"
                : entry.type().startsWith("CANVAS_RUN_") ? "CANVAS_RUN"
                : entry.type().startsWith("COMFY_TEST_") || entry.type().startsWith("CANVAS_MEDIA_") || entry.type()
                .startsWith("MEDIA_IMPORT_") || entry.type().startsWith("MEDIA_PREVIEW_") || entry.type()
                .equals("EPISODE_AUDIO_REVIEW")
                ? "MEDIA" : "GENERATION";
        entries.appendEvent(eventStream, entry.id(), entry.aggregate(), entry.payload().path("revision").asLong(),
                entry.type(), json.write(entry.payload()));
        entries.delivered(entry.id());
    }

    public static Map<String, Object> state(String id, long revision, String state) {
        return Map.of("submissionId", id, "revision", revision, "state", state);
    }

    private record Entry(long id, String type, String aggregate, ObjectNode payload) {
    }

    private static final class DeliveryFailure extends RuntimeException {
        private final long id;

        DeliveryFailure(long id, RuntimeException cause) {
            super(cause);
            this.id = id;
        }
    }
}
