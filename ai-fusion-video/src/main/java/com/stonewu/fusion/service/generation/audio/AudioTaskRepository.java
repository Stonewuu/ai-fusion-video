package com.stonewu.fusion.service.generation.audio;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.mapper.generation.AudioTaskMapper;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.entity.ai.AiModel;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

@Repository
@RequiredArgsConstructor
public class AudioTaskRepository {
    private final AudioTaskMapper tasks;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;
    private final AiModelMapper models;

    @Transactional
    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public void insert(AudioTask task) {
        LocalDateTime now = coordination.now();
        task.setCreateTime(now);
        task.setUpdateTime(now);
        task.setNextPollAt(now);
        task.setDeadlineAt(now.plusMinutes(15));
        tasks.insert(task);
    }

    public AudioTask findRequest(long userId, String requestId) {
        return tasks.selectOne(new LambdaQueryWrapper<AudioTask>()
                .eq(AudioTask::getUserId, userId).eq(AudioTask::getRequestId, requestId));
    }

    public AudioTask owned(String id, CanvasActor actor) {
        var task = tasks.selectById(id);
        if (task == null || task.getUserId() != actor.userId() || !Objects.equals(task.getTeamId(), actor.teamId())) {
            throw new BusinessException(404, "音频生成记录不存在");
        }
        return task;
    }

    public List<AudioTask> recent(CanvasActor actor) {
        var ids = tasks.selectPage(new Page<>(1, 30, false), new LambdaQueryWrapper<AudioTask>()
                .select(AudioTask::getId)
                .eq(AudioTask::getUserId, actor.userId())
                .eq(actor.teamId() != null, AudioTask::getTeamId, actor.teamId())
                .isNull(actor.teamId() == null, AudioTask::getTeamId)
                .orderByDesc(AudioTask::getCreateTime)).getRecords().stream().map(AudioTask::getId).toList();
        if (ids.isEmpty()) {
            return List.of();
        }
        var records = tasks.selectByIds(ids).stream().collect(Collectors.toMap(AudioTask::getId, Function.identity()));
        return ids.stream().map(records::get).toList();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public Optional<AudioTask> claim(String owner) {
        coordination.lockGenerationDispatch();
        LocalDateTime now = coordination.now();
        // 未收到结果时不重发付费请求；已经接收的响应可以继续解析和保存。
        tasks.update(null, new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getState, "RUNNING")
                .lt(AudioTask::getLeaseUntil, now)
                .isNull(AudioTask::getOutputJson)
                .set(AudioTask::getState, "UNKNOWN")
                .set(AudioTask::getErrorMessage, "调用已中断，结果待确认，请检查 Google 记录后再试"));
        tasks.update(null, new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getState, "RUNNING")
                .lt(AudioTask::getLeaseUntil, now).isNotNull(AudioTask::getOutputJson)
                .set(AudioTask::getState, "INGESTING").set(AudioTask::getNextPollAt, now));
        var candidates = tasks.selectPage(new Page<>(1, 20, false), new LambdaQueryWrapper<AudioTask>()
                .select(AudioTask::getId)
                .in(AudioTask::getState, "QUEUED", "INGESTING").le(AudioTask::getNextPollAt, now)
                .orderByAsc(AudioTask::getNextPollAt)).getRecords();
        for (var candidate : candidates) {
            var task = tasks.selectById(candidate.getId());
            if (task.getDeadlineAt().isBefore(now)) {
                tasks.update(null, new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getId, task.getId())
                        .set(AudioTask::getState, "FAILED")
                        .set(AudioTask::getErrorMessage, "任务超过等待时限，请到 Google 查看结果"));
                continue;
            }
            AiModel model = models.selectById(task.getModelId());
            if ("QUEUED".equals(task.getState()) && (model == null || !Integer.valueOf(1).equals(model.getStatus()))) {
                tasks.update(null, new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getId, task.getId())
                        .set(AudioTask::getState, "FAILED").set(AudioTask::getErrorMessage, "模型已停用，请选择其他模型"));
                continue;
            }
            if ("QUEUED".equals(task.getState()) && model != null && model.getMaxConcurrency() != null && model.getMaxConcurrency() > 0) {
                long active = tasks.selectCount(new LambdaQueryWrapper<AudioTask>()
                        .eq(AudioTask::getModelId, task.getModelId()).isNull(AudioTask::getOutputJson)
                        .eq(AudioTask::getState, "RUNNING"));
                if (active >= model.getMaxConcurrency()) {
                    tasks.update(null, new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getId, task.getId())
                            .set(AudioTask::getNextPollAt, now.plusSeconds(1)));
                    continue;
                }
            }
            tasks.update(null, new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getId, task.getId())
                    .set(AudioTask::getState, "RUNNING").set(AudioTask::getLeaseOwner, owner)
                    .set(AudioTask::getLeaseUntil, now.plusMinutes(2)).set(AudioTask::getUpdateTime, now));
            task.setLeaseOwner(owner);
            task.setState("RUNNING");
            return Optional.of(task);
        }
        return Optional.empty();
    }

    public void assertActive(AudioTask task) {
        var current = tasks.selectById(task.getId());
        if (current == null || !"RUNNING".equals(current.getState())
                || !Objects.equals(task.getLeaseOwner(), current.getLeaseOwner())
                || current.getLeaseUntil().isBefore(coordination.now())) {
            throw new BusinessException(409, "任务执行已停止");
        }
    }

    public void heartbeat(AudioTask task) {
        tasks.update(null, active(task).set(AudioTask::getLeaseUntil, coordination.now().plusMinutes(2)));
    }

    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public void received(AudioTask task, AudioTaskData.Received output) {
        var now = coordination.now();
        tasks.update(null, active(task).set(AudioTask::getState, "INGESTING")
                .set(AudioTask::getOutputJson, json.write(output)).set(AudioTask::getNextPollAt, now)
                .set(AudioTask::getLeaseOwner, null).set(AudioTask::getLeaseUntil, null)
                .set(AudioTask::getUpdateTime, now));
    }

    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public void retrySave(String id, CanvasActor actor) {
        owned(id, actor);
        int changed = tasks.update(null, new LambdaUpdateWrapper<AudioTask>()
                .eq(AudioTask::getId, id).eq(AudioTask::getState, "FAILED")
                .isNotNull(AudioTask::getOutputJson).set(AudioTask::getState, "INGESTING")
                .set(AudioTask::getNextPollAt, coordination.now())
                .set(AudioTask::getDeadlineAt, coordination.now().plusMinutes(15))
                .set(AudioTask::getErrorMessage, null));
        if (changed != 1) {
            throw new BusinessException("当前任务没有待保存的输出，请刷新后重试");
        }
    }

    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public void finish(AudioTask task, String state, AudioTaskData.Result result, String error) {
        tasks.update(null, active(task).set(AudioTask::getState, state)
                .set("COMPLETED".equals(state), AudioTask::getOutputJson, null)
                .set(AudioTask::getResultJson, result == null ? null : json.write(result))
                .set(AudioTask::getErrorMessage, error).set(AudioTask::getLeaseOwner, null)
                .set(AudioTask::getLeaseUntil, null).set(AudioTask::getUpdateTime, coordination.now()));
    }

    private LambdaUpdateWrapper<AudioTask> active(AudioTask task) {
        return new LambdaUpdateWrapper<AudioTask>().eq(AudioTask::getId, task.getId())
                .eq(AudioTask::getState, "RUNNING").eq(AudioTask::getLeaseOwner, task.getLeaseOwner())
                .ge(AudioTask::getLeaseUntil, coordination.now());
    }
}
