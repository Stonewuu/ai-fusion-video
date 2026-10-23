package com.stonewu.fusion.service.canvas.template;

import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources.Entry;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.model.MediaFacts;

import java.util.List;
import java.util.Map;

/**
 * 模板编号只在文件内部有意义；导入必须重新解析业务编号、提供商和媒体绑定。
 */
public final class CanvasTemplateData {
    private CanvasTemplateData() {
    }

    public static final String SCHEMA = "afv.canvas.template";
    public static final int VERSION = 1;
    public static final int MAX_BYTES = 16 * 1024 * 1024;

    public enum ProviderKind {TEXT_MODEL, IMAGE_MODEL, VIDEO_MODEL, COMFYUI_MODEL, AUDIO_API, WORKFLOW, MCP}

    public record Use(String nodeId, boolean batch, String field) {
    }

    public record Provider(String id, ProviderKind kind, String label, List<Use> uses) {
    }

    public record Media(String id, String name, MediaFacts content) {
    }

    public record Template(String schema, int schemaVersion, String title, CanvasGraph graph, List<Entry> objects,
                           Long primaryScript, Long primaryStoryboard, List<Media> media, List<Provider> providers,
                           Map<String, Value> fixedOutputs) {
    }

    public record Binding(String id, Long version) {
    }

    public record Bindings(Map<String, Binding> media, Map<String, Binding> providers) {
    }
}
