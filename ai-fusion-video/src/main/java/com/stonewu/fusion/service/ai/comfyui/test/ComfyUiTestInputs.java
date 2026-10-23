package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiInputBinding;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowDocumentService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowRenderer;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class ComfyUiTestInputs {
    private final ComfyUiWorkflowDocumentService documents;
    private final ComfyUiWorkflowRenderer renderer;
    private final MediaResourceQueries media;
    private final MediaInputTransport transport;

    public ComfyUiTestSnapshot freeze(ComfyUiWorkflow workflow, ComfyUiWorkflowVersion version,
                                      Map<String, Object> inputs, CanvasActor actor) {
        var bindings = bindings(workflow, version);
        Set<String> known = bindings.stream().map(ComfyUiInputBinding::businessField).collect(Collectors.toSet());
        if (!known.containsAll(inputs.keySet())) {
            throw MediaException.invalid("试运行输入含未绑定字段，请刷新工作流版本");
        }
        renderer.render(workflow.getModelType(), version, inputs);
        Map<String, Integer> versions = new LinkedHashMap<>();
        for (String field : imageFields(bindings)) {
            for (String source : sources(inputs.get(field))) {
                if (!MediaReference.managed(source)) {
                    continue;
                }
                var resource = media.get(MediaReference.id(source), actor);
                if (resource.content().kind() != MediaKind.IMAGE) {
                    throw MediaException.invalid("试运行上传绑定必须使用图片");
                }
                versions.put(resource.id(), resource.version());
            }
        }
        return new ComfyUiTestSnapshot(1, workflow, version, inputs, Map.copyOf(versions));
    }

    public Map<String, Object> resolve(ComfyUiTestSnapshot snapshot, CanvasActor actor) {
        Map<String, Object> result = new LinkedHashMap<>(snapshot.inputs());
        for (String field : imageFields(bindings(snapshot.workflow(), snapshot.version()))) {
            if (!result.containsKey(field)) {
                continue;
            }
            result.put(field, sources(result.get(field)).stream().map(source -> {
                if (!MediaReference.managed(source)) {
                    return source;
                }
                String id = MediaReference.id(source);
                var resource = media.get(id, actor);
                if (!Integer.valueOf(resource.version()).equals(snapshot.mediaVersions().get(id))) {
                    throw new MediaException(409, "COMFY_TEST_MEDIA_CHANGED", "试运行引用的输入媒体版本已变化");
                }
                return transport.dataUri(source, actor, MediaKind.IMAGE, 20L * 1024 * 1024);
            }).toList());
        }
        return result;
    }

    private List<ComfyUiInputBinding> bindings(ComfyUiWorkflow workflow, ComfyUiWorkflowVersion version) {
        return documents.parseInputBindings(workflow.getModelType(), version.getApiWorkflowJson(),
                version.getInputBindingsJson());
    }

    private Set<String> imageFields(List<ComfyUiInputBinding> bindings) {
        if (bindings.stream().anyMatch(binding -> binding.valueType().equals("uploaded_video") || binding.valueType()
                .equals("uploaded_audio"))) {
            throw MediaException.invalid("当前 ComfyUI Native API 尚不支持视频或音频上传绑定");
        }
        return bindings.stream().filter(binding -> binding.valueType().equals("uploaded_image"))
                .map(ComfyUiInputBinding::businessField).collect(Collectors.toSet());
    }

    private List<String> sources(Object value) {
        if (value == null) {
            return List.of();
        }
        var values = value instanceof List<?> list ? list : List.of(value);
        if (values.size() > 64) {
            throw MediaException.invalid("单个试运行输入集合不能超过 64 项");
        }
        return values.stream().map(item -> {
            if (!(item instanceof String source) || source.isBlank()
                    || !(MediaReference.managed(source) || source.startsWith("https://")
                    || source.startsWith("http://") || source.startsWith("data:image/"))) {
                throw MediaException.invalid("图片输入须使用已授权媒体、图片 Data URI 或 HTTP(S) 地址");
            }
            return source;
        }).toList();
    }
}
