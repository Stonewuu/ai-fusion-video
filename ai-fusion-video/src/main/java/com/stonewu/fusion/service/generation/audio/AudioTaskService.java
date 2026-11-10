package com.stonewu.fusion.service.generation.audio;

import cn.hutool.core.bean.BeanUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.google.GoogleModelFamily;
import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Transactional(propagation = Propagation.NEVER)
public class AudioTaskService {
    private final AudioTaskRepository tasks;
    private final AiModelService models;
    private final ApiConfigService configs;
    private final MediaResourceQueries media;
    private final CanvasJson json;
    private final GenerationModelCapabilityService capabilities;

    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public AudioTaskData.View create(AudioTaskData.Create request, CanvasActor actor) {
        String hash = json.hash(request);
        var existing = tasks.findRequest(actor.userId(), request.requestId());
        if (existing != null) {
            return repeated(existing, hash, actor);
        }
        var model = models.getById(request.modelId());
        if (model == null || !Integer.valueOf(1).equals(model.getStatus())) {
            throw new BusinessException("请选择已启用的模型");
        }
        var api = configs.getById(model.getApiConfigId());
        if (api == null || !Integer.valueOf(1).equals(api.getStatus()) || !GoogleModelFamily.googlePlatform(api.getPlatform())) {
            throw new BusinessException("请选择已启用的 Google 渠道");
        }
        Integer type = GoogleModelFamily.type(model.getCode());
        if (!Integer.valueOf(4).equals(type) || !type.equals(model.getModelType())) {
            throw new BusinessException("请选择音频生成模型");
        }
        var kinds = new ArrayList<MediaKind>();
        for (String reference : request.input().media()) {
            var resource = media.get(MediaReference.id(reference), actor);
            if (!"READY".equals(resource.state())) {
                throw new BusinessException("参考素材尚未就绪，请重新选择");
            }
            kinds.add(resource.content().kind());
        }
        validateInput(model.getCode(), kinds);
        var task = new AudioTask();
        task.setId(UUID.randomUUID().toString());
        task.setRequestId(request.requestId());
        task.setUserId(actor.userId());
        task.setTeamId(actor.teamId());
        task.setModelId(model.getId());
        task.setModelName(model.getName());
        var snapshot = BeanUtil.copyProperties(model, AiModel.class);
        snapshot.setConfig(capabilities.getMergedModelConfig(model).toString());
        snapshot.setCapabilityPresetCode(null);
        task.setModelSnapshotJson(json.write(snapshot));
        task.setApiConfigId(model.getApiConfigId());
        task.setApiRouteJson(json.write(new AudioTaskData.Route(api.getPlatform(), api.getApiUrl(), api.getAppId())));
        task.setInputJson(json.write(request.input()));
        task.setInputHash(hash);
        task.setState("QUEUED");
        try {
            tasks.insert(task);
        } catch (DuplicateKeyException duplicate) {
            return repeated(tasks.findRequest(actor.userId(), request.requestId()), hash, actor);
        }
        return view(task);
    }

    static void validateInput(String code, List<MediaKind> kinds) {
        if (!code.startsWith("lyria-3") && !kinds.isEmpty()) {
            throw new BusinessException("此音频模型使用文字输入，请移除参考素材");
        }
        if (kinds.stream().anyMatch(kind -> kind != MediaKind.IMAGE)) {
            throw new BusinessException("请选择图片作为音乐参考");
        }
    }

    @Cacheable(value = "audioGenerationTasks", key = "#id + ':' + #actor.userId() + ':' + #actor.teamId()")
    public AudioTaskData.View get(String id, CanvasActor actor) {
        return view(tasks.owned(id, actor));
    }

    public List<AudioTaskData.View> list(CanvasActor actor) {
        return tasks.recent(actor).stream().map(this::view).toList();
    }

    @CacheEvict(value = "audioGenerationTasks", allEntries = true)
    public void retrySave(String id, CanvasActor actor) {
        tasks.retrySave(id, actor);
    }

    private AudioTaskData.View repeated(AudioTask task, String hash, CanvasActor actor) {
        if (task == null || !Objects.equals(task.getTeamId(), actor.teamId()) || !hash.equals(task.getInputHash())) {
            throw new BusinessException(409, "请求内容已改变，请重新发起");
        }
        return view(task);
    }

    private AudioTaskData.View view(AudioTask task) {
        return new AudioTaskData.View(task.getId(), task.getModelId(), task.getModelName(),
                task.getState(), task.getResultJson() == null ? null
                : json.read(task.getResultJson(), AudioTaskData.Result.class),
                task.getErrorMessage(), "FAILED".equals(task.getState()) && task.getOutputJson() != null,
                task.getCreateTime(), task.getUpdateTime());
    }
}
