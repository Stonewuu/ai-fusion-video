package com.stonewu.fusion.service.generation.audio;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.ai.google.GoogleNativeApi;
import com.stonewu.fusion.service.ai.google.GoogleLiveMusicClient;
import com.stonewu.fusion.service.ai.google.GoogleAudioProtocol;
import com.stonewu.fusion.service.ai.google.GoogleAudio;
import com.stonewu.fusion.service.ai.google.GoogleModelFamily;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaIoBoundary;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class AudioTaskRunner {
    private final GoogleNativeApi google;
    private final GoogleLiveMusicClient live;
    private final ApiConfigService configs;
    private final GenerationModelCapabilityService capabilities;
    private final MediaInputTransport inputs;
    private final MediaResourceQueries resources;
    private final MediaIntakeService intake;
    private final CanvasJson json;
    private final AudioTaskRepository tasks;

    public AudioTaskData.Received submit(AudioTask task) {
        MediaIoBoundary.requireOutsideTransaction();
        AiModel model = json.read(task.getModelSnapshotJson(), AiModel.class);
        var config = capabilities.getMergedModelConfig(model);
        var api = api(task, config);
        var input = json.read(task.getInputJson(), AudioTaskData.Input.class);
        var actor = new CanvasActor(task.getUserId(), task.getTeamId());
        var sources = new ArrayList<String>();
        for (String reference : input.media()) {
            var resource = resources.get(MediaReference.id(reference), actor);
            sources.add(inputs.dataUri(reference, actor, resource.content().kind(), MediaIntakeService.MAX_UPLOAD_BYTES));
        }
        tasks.assertActive(task);
        if (model.getCode().startsWith("lyria-realtime")) {
            var response = live.generate(model, api, input.text(), config, () -> tasks.assertActive(task));
            return new AudioTaskData.Received("normalized", json.write(response));
        }
        var spec = GoogleAudioProtocol.request(model, input, sources, config, google.developerApi(api));
        String endpoint = "interactions".equals(spec.operation()) ? google.interactionsUrl(api)
                : google.modelUrl(api, model.getCode(), spec.operation());
        var request = google.post(api, endpoint, spec.body());
        tasks.assertActive(task);
        var response = google.execute(api, request);
        return new AudioTaskData.Received(spec.operation(), response.toString());
    }


    private ApiConfig api(AudioTask task, JSONObject config) {
        var original = configs.getById(task.getApiConfigId());
        if (original == null || !Integer.valueOf(1).equals(original.getStatus())
                || !GoogleModelFamily.googlePlatform(original.getPlatform())) {
            throw new BusinessException("Google 渠道已停用，请检查 API 配置");
        }
        var route = json.read(task.getApiRouteJson(), AudioTaskData.Route.class);
        if (!route.equals(new AudioTaskData.Route(original.getPlatform(), original.getApiUrl(), original.getAppId()))) {
            throw new BusinessException("任务的 API 配置已变更，请重新发起");
        }
        var target = BeanUtil.copyProperties(original, ApiConfig.class);
        if (!google.developerApi(target) && config.containsKey("googleLocation")) {
            target.setApiUrl(config.getStr("googleLocation"));
        }
        return target;
    }

    public AudioTaskData.Result save(AudioTask task) {
        MediaIoBoundary.requireOutsideTransaction();
        var received = json.read(task.getOutputJson(), AudioTaskData.Received.class);
        var output = "normalized".equals(received.operation()) ? json.read(received.body(), GoogleAudioProtocol.Output.class)
                : GoogleAudioProtocol.output(JSONUtil.parseObj(received.body()), received.operation());
        var audio = new ArrayList<String>();
        var actor = new CanvasActor(task.getUserId(), task.getTeamId());
        for (int index = 0; index < output.audio().size(); index++) {
            tasks.assertActive(task);
            var part = output.audio().get(index);
            if (part.data() == null || part.mime() == null || !part.mime().startsWith("audio/")) {
                throw new GoogleAudioProtocol.InvalidOutput("Google 音频内容无法读取");
            }
            byte[] bytes = Base64.getDecoder().decode(part.data());
            String mime = part.mime();
            if (mime.toLowerCase().contains("l16") || mime.toLowerCase().contains("pcm")) {
                bytes = GoogleAudio.wave(bytes, mime);
                mime = "audio/wav";
            }
            String extension = mime.contains("mpeg") || mime.contains("mp3") ? ".mp3" : ".wav";
            try {
                var resource = intake.uploadChecked(new ByteArrayInputStream(bytes), "生成音频" + extension, mime,
                        task.getId() + "-audio-" + index, null, ResourceCommandContext.user(actor, null), Set.of(MediaKind.AUDIO));
                audio.add(MediaReference.of(resource.id()));
            } catch (IOException failure) {
                throw new UncheckedIOException("生成音频保存失败", failure);
            }
        }
        tasks.assertActive(task);
        return new AudioTaskData.Result(output.text(), audio);
    }

}
