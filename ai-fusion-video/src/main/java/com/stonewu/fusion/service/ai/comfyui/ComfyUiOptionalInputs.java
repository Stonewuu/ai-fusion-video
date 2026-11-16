package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 可选媒体缺省时移除指定连接，并清理不再参与输出的节点。 */
final class ComfyUiOptionalInputs {
    private ComfyUiOptionalInputs() {
    }

    record Target(String nodeId, String inputName) {
        String label() {
            return nodeId + "." + inputName;
        }
    }

    record Plan(Set<String> nodes, Set<Target> inputs) {
    }

    static boolean isMissing(ComfyUiInputBinding binding, Map<String, Object> values) {
        Object raw = values == null ? null : values.get(binding.businessField());
        if (raw == null) {
            return true;
        }
        if (raw instanceof List<?> list) {
            int index = binding.index() == null ? 0 : binding.index();
            return index >= list.size() || list.get(index) == null;
        }
        return false;
    }

    static boolean isUploaded(String type) {
        return Set.of("uploaded_image", "uploaded_audio", "uploaded_video").contains(type);
    }

    static boolean isLink(JsonNode value) {
        return value != null && value.isArray() && value.size() == 2
                && value.get(0).isTextual() && value.get(1).isInt();
    }

    static Target uploadTarget(ObjectNode workflow, ComfyUiInputBinding binding) {
        Target target = new Target(binding.nodeId(), binding.inputName());
        JsonNode current = workflow.path(target.nodeId()).path("inputs").get(target.inputName());
        if (!isUploaded(binding.valueType()) || !isLink(current)) {
            return target;
        }
        JsonNode source = workflow.path(current.get(0).asText());
        String expectedClass = switch (binding.valueType()) {
            case "uploaded_image" -> "LoadImage";
            case "uploaded_audio" -> "LoadAudio";
            case "uploaded_video" -> "LoadVideo";
            default -> throw new IllegalArgumentException("不是上传媒体绑定");
        };
        String fileInput = switch (binding.valueType()) {
            case "uploaded_image" -> "image";
            case "uploaded_audio" -> "audio";
            case "uploaded_video" -> "file";
            default -> throw new IllegalArgumentException("不是上传媒体绑定");
        };
        if (current.get(1).asInt() != 0 || !expectedClass.equals(source.path("class_type").asText())
                || !source.path("inputs").path(fileInput).isTextual()) {
            throw new BusinessException(400,
                    "上传媒体请直接绑定加载节点的文件输入；连线仅支持 LoadImage 的 IMAGE 输出、LoadAudio 的 AUDIO 输出或 LoadVideo 的 VIDEO 输出: "
                            + target.label());
        }
        return new Target(current.get(0).asText(), fileInput);
    }

    static Plan plan(ObjectNode workflow, List<ComfyUiInputBinding> omitted) {
        Set<String> nodes = new LinkedHashSet<>();
        Set<Target> inputs = new LinkedHashSet<>();
        for (ComfyUiInputBinding binding : omitted) {
            JsonNode current = workflow.path(binding.nodeId()).path("inputs").get(binding.inputName());
            if (isUploaded(binding.valueType()) && !isLink(current)) {
                nodes.add(binding.nodeId());
            } else {
                inputs.add(new Target(binding.nodeId(), binding.inputName()));
            }
        }
        workflow.fields().forEachRemaining(node -> node.getValue().path("inputs").fields().forEachRemaining(input -> {
            if (isLink(input.getValue()) && nodes.contains(input.getValue().get(0).asText())) {
                inputs.add(new Target(node.getKey(), input.getKey()));
            }
        }));
        return new Plan(Set.copyOf(nodes), Set.copyOf(inputs));
    }

    static void apply(ObjectNode workflow, Plan plan, List<ComfyUiOutputBinding> outputs) {
        Set<String> originalDependencies = dependencies(workflow, outputs);
        for (ComfyUiOutputBinding output : outputs) {
            if (plan.nodes().contains(output.nodeId())) {
                throw new BusinessException(400, "可选输入不能移除已配置的输出节点: " + output.nodeId());
            }
        }
        for (Target input : plan.inputs()) {
            ((ObjectNode) workflow.path(input.nodeId()).path("inputs")).remove(input.inputName());
        }
        workflow.remove(plan.nodes());
        Set<String> remainingDependencies = dependencies(workflow, outputs);
        List<String> detached = new ArrayList<>();
        workflow.fieldNames().forEachRemaining(id -> {
            if (originalDependencies.contains(id) && !remainingDependencies.contains(id)) {
                detached.add(id);
            }
        });
        workflow.remove(detached);
    }

    private static Set<String> dependencies(ObjectNode workflow, List<ComfyUiOutputBinding> outputs) {
        Set<String> reachable = new HashSet<>();
        var pending = new ArrayDeque<String>();
        outputs.forEach(output -> pending.add(output.nodeId()));
        while (!pending.isEmpty()) {
            String id = pending.removeFirst();
            if (!reachable.add(id)) {
                continue;
            }
            workflow.path(id).path("inputs").forEach(input -> {
                if (isLink(input)) {
                    pending.addLast(input.get(0).asText());
                }
            });
        }
        return reachable;
    }

    static void requirePreservedBindings(ObjectNode rendered, Plan plan,
                                         List<ComfyUiInputBinding> bindings, List<ComfyUiInputBinding> omitted) {
        for (ComfyUiInputBinding binding : bindings) {
            if (omitted.contains(binding)) {
                continue;
            }
            Target target = new Target(binding.nodeId(), binding.inputName());
            if (!rendered.has(target.nodeId()) || plan.inputs().contains(target)) {
                throw new BusinessException(400,
                        "省略可选输入会移除其他绑定 " + binding.businessField() + "（" + target.label()
                                + "），请分别连接加载节点");
            }
        }
    }
}
