package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static com.stonewu.fusion.service.media.importing.MediaImportData.*;

@Service
@RequiredArgsConstructor
public class MediaImportJobs {
    private final MediaJobMapper mediaJobs;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final MediaImportAccess access;
    private final MediaImportService service;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;
    private final MediaResourceQueries media;

    public record Work(MediaJob job, int index, Source source) {
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<MediaJob> claim(String owner) {
        coordination.lockGenerationDispatch();
        if (mediaJobs.activeImports() >= 4) {
            return Optional.empty();
        }
        var ids = mediaJobs.nextImport();
        if (ids.isEmpty()) {
            return Optional.empty();
        }
        String id = ids.getFirst();
        mediaJobs.claimCompose(owner, id);
        return Optional.of(jobs.get(id));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public Work start(MediaJob.Claim claim) {
        var job = checked(claim);
        var checkpoint = access.checkpoint(job);
        var items = new ArrayList<>(checkpoint.items());
        for (int index = 0; index < items.size(); index++) {
            var item = items.get(index);
            if (item.status().equals("PENDING") || item.status().equals("RUNNING")) {
                items.set(index, new Item("RUNNING", item.attempt(), null, null, null, null));
                var source = access.input(job).sources().get(index);
                mediaJobs.checkpoint(json.write(new Checkpoint(1, List.copyOf(items), checkpoint.lastRetryHash())),
                        "正在导入第 " + (index + 1) + " / " + items.size() + " 项", job.id());
                service.event(jobs.get(job.id()), "MEDIA_IMPORT_CHANGED", Map.of("index", index));
                return new Work(jobs.get(job.id()), index, source);
            }
        }
        saveProgress(job, checkpoint);
        return null;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public MediaJob checked(MediaJob.Claim claim) {
        var job = jobs.get(claim.id());
        if (!KIND.equals(job.kind())) {
            throw MediaJobFence.lost();
        }
        fence.require(claim, job.context(), false);
        return jobs.get(job.id());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public void record(MediaJob.Claim claim, int index, String resourceId, String errorCode, String message) {
        var job = checked(claim);
        var checkpoint = access.checkpoint(job);
        var items = new ArrayList<>(checkpoint.items());
        if (index < 0 || index >= items.size() || !items.get(index).status().equals("RUNNING")) {
            throw MediaJobFence.lost();
        }
        if (resourceId != null && !media.get(resourceId, job.context().actor()).state().equals("READY")) {
            throw MediaException.invalid("导入结果尚未登记完成");
        }
        var item = items.get(index);
        items.set(index,
                new Item(resourceId == null ? "FAILED" : "SUCCEEDED", item.attempt(), resourceId, errorCode, message,
                        null));
        saveProgress(job, new Checkpoint(1, List.copyOf(items), checkpoint.lastRetryHash()));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public void waitForIngest(MediaJob.Claim claim) {
        var job = checked(claim);
        mediaJobs.waitForImportIngest(job.id());
        service.event(jobs.get(job.id()), "MEDIA_IMPORT_CHANGED", Map.of());
    }

    /**
     * 撤权或取消只结束当前租约自己的记录，不发布结果、不修改画布或业务内容。
     */
    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public void stop(MediaJob.Claim claim, String message) {
        var ids = mediaJobs.lockImportOwner(claim.id(), claim.owner(), claim.epoch());
        if (ids.isEmpty()) {
            return;
        }
        var job = jobs.get(claim.id());
        var checkpoint = access.checkpoint(job);
        var items = new ArrayList<>(checkpoint.items());
        MediaImportService.cancelUnfinished(items);
        mediaJobs.stopImport(message, json.write(new Checkpoint(1, List.copyOf(items), checkpoint.lastRetryHash())),
                job.id());
        service.event(jobs.get(job.id()), "MEDIA_IMPORT_CANCELLED", Map.of("reason", message));
    }

    private void saveProgress(MediaJob job, Checkpoint checkpoint) {
        boolean complete = checkpoint.items().stream().allMatch(item -> MediaImportQueries.terminal(item.status()));
        long succeeded = checkpoint.items().stream().filter(item -> item.status().equals("SUCCEEDED")).count();
        String status = !complete ? "WAITING" : succeeded == checkpoint.items()
                .size() ? "SUCCEEDED" : succeeded == 0 ? "FAILED" : "PARTIAL_FAILED";
        String phase = !complete ? "继续导入剩余素材" : status.equals(
                "SUCCEEDED") ? "全部素材已导入" : "导入已结束，部分素材需要处理";
        mediaJobs.importProgress(status, json.write(checkpoint), phase, job.id());
        service.event(jobs.get(job.id()), "MEDIA_IMPORT_CHANGED", Map.of("completed", complete));
    }
}
