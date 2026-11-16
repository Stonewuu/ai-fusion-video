package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiInputBinding;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowDocumentService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowRenderer;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Locale;
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
        for (var field : mediaFields(bindings).entrySet()) {
            for (String source : sources(inputs.get(field.getKey()), field.getValue())) {
                if (!MediaReference.managed(source)) {
                    continue;
                }
                var resource = media.get(MediaReference.id(source), actor);
                if (resource.content().kind() != field.getValue()) {
                    throw MediaException.invalid("试运行输入媒体类型与绑定不匹配");
                }
                versions.put(resource.id(), resource.version());
            }
        }
        return new ComfyUiTestSnapshot(1, workflow, version, inputs, Map.copyOf(versions));
    }

    public Map<String, Object> resolve(ComfyUiTestSnapshot snapshot, CanvasActor actor) {
        Map<String, Object> result = new LinkedHashMap<>(snapshot.inputs());
        for (var field : mediaFields(bindings(snapshot.workflow(), snapshot.version())).entrySet()) {
            if (!result.containsKey(field.getKey())) {
                continue;
            }
            result.put(field.getKey(), sources(result.get(field.getKey()), field.getValue()).stream().map(source -> {
                if (!MediaReference.managed(source)) {
                    return source;
                }
                String id = MediaReference.id(source);
                var resource = media.get(id, actor);
                if (!Integer.valueOf(resource.version()).equals(snapshot.mediaVersions().get(id))) {
                    throw new MediaException(409, "COMFY_TEST_MEDIA_CHANGED", "试运行引用的输入媒体版本已变化");
                }
                long limit = field.getValue() == MediaKind.IMAGE ? 20L * 1024 * 1024 : MediaIntakeService.MAX_UPLOAD_BYTES;
                return transport.dataUri(source, actor, field.getValue(), limit);
            }).toList());
        }
        return result;
    }

    private List<ComfyUiInputBinding> bindings(ComfyUiWorkflow workflow, ComfyUiWorkflowVersion version) {
        return documents.parseInputBindings(workflow.getModelType(), version.getApiWorkflowJson(),
                version.getInputBindingsJson());
    }

    private Map<String, MediaKind> mediaFields(List<ComfyUiInputBinding> bindings) {
        Map<String, MediaKind> result = new LinkedHashMap<>();
        for (var binding : bindings) {
            MediaKind kind = switch (binding.valueType()) {
                case "uploaded_image" -> MediaKind.IMAGE;
                case "uploaded_video" -> MediaKind.VIDEO;
                case "uploaded_audio" -> MediaKind.AUDIO;
                default -> null;
            };
            if (kind != null) {
                var previous = result.putIfAbsent(binding.businessField(), kind);
                if (previous != null && previous != kind) {
                    throw MediaException.invalid("同一工作流输入不能绑定多种媒体类型");
                }
            }
        }
        return result;
    }

    private List<String> sources(Object value, MediaKind kind) {
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
                    || source.startsWith("http://") || source.startsWith("data:" + kind.name().toLowerCase(Locale.ROOT) + "/"))) {
                throw MediaException.invalid("输入须使用已授权媒体、对应类型的 Data URI 或 HTTP(S) 地址");
            }
            return source;
        }).toList();
    }
}
