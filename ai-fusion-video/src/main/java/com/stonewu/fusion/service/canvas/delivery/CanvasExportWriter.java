package com.stonewu.fusion.service.canvas.delivery;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.MediaContentReader;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static com.stonewu.fusion.service.canvas.delivery.CanvasExportData.*;

@Component
@RequiredArgsConstructor
public class CanvasExportWriter {
    private final CanvasJson json;
    private final MediaContentReader contents;
    private static final Map<String, String> LABELS = Map.ofEntries(
            Map.entry("id", "编号"), Map.entry("version", "版本"), Map.entry("name", "名称"),
            Map.entry("title", "标题"),
            Map.entry("summary", "概要"), Map.entry("rawContent", "原文"), Map.entry("content", "内容"),
            Map.entry("sceneHeading", "场景标题"), Map.entry("sceneDescription", "场景描述"),
            Map.entry("dialogues", "对白"),
            Map.entry("description", "描述"), Map.entry("visualDescription", "画面描述"),
            Map.entry("shotNumber", "镜头编号"),
            Map.entry("duration", "时长"), Map.entry("shotType", "景别"), Map.entry("cameraMovement", "运镜"),
            Map.entry("videoPrompt", "视频提示词"), Map.entry("firstFramePrompt", "首帧提示词"),
            Map.entry("lastFramePrompt", "尾帧提示词"),
            Map.entry("firstFrameImageUrl", "首帧"), Map.entry("lastFrameImageUrl", "尾帧"),
            Map.entry("videoUrl", "视频"),
            Map.entry("generatedVideoUrl", "生成视频"), Map.entry("sortOrder", "排序"),
            Map.entry("properties", "扩展属性"),
            Map.entry("scriptEpisodeId", "剧本分集"), Map.entry("scriptSceneId", "剧本场次"),
            Map.entry("episodeId", "分集编号"),
            Map.entry("storyboardSceneId", "分镜场次编号"), Map.entry("storyboardId", "分镜编号"));

    public void write(Manifest manifest, Format format, CanvasActor actor, Path target,
                      Runnable check) throws IOException {
        check.run();
        switch (format) {
            case MARKDOWN -> Files.writeString(target, markdown(manifest), StandardCharsets.UTF_8);
            case CSV -> Files.writeString(target, csv(manifest), StandardCharsets.UTF_8);
            case JSON -> Files.writeString(target, document(manifest), StandardCharsets.UTF_8);
            case MEDIA -> contents.copy(manifest.media().getFirst().id(), actor, target, MAX_BYTES, check);
            case ZIP -> zip(manifest, actor, target, check);
        }
        check.run();
        if (Files.size(target) > MAX_BYTES + 32L * 1024 * 1024) {
            throw invalid("交付文件超过大小限制，请缩小范围");
        }
    }

    public String document(Manifest manifest) {
        var paths = new LinkedHashMap<String, String>();
        manifest.media().forEach(file -> paths.put(file.id(), mediaPath(file)));
        return json.write(Map.of("schema", "afv.canvas.delivery", "schemaVersion", manifest.schemaVersion(),
                "snapshot", manifest, "mediaFiles", paths));
    }

    public String markdown(Manifest manifest) {
        StringBuilder out = new StringBuilder("# ").append(plain(manifest.title())).append("\n\n")
                .append("画布版本：").append(manifest.documentRevision()).append("\n\n");
        for (Value value : manifest.values()) {
            appendValue(out, value, "##");
        }
        if (!manifest.fixedOutputs().isEmpty()) {
            out.append("## 固定连线结果\n\n此处保留连线选定时的内容，上游后续修改不会替换这些结果。\n\n");
            for (var edge : manifest.graph().edges()) {
                var value = manifest.fixedOutputs().get(edge.id());
                if (value == null) {
                    continue;
                }
                out.append("### ").append(nodeTitle(manifest, edge.sourceNodeId())).append(" → ")
                        .append(nodeTitle(manifest, edge.targetNodeId())).append("\n\n")
                        .append("连线：").append(edge.id()).append("\n\n");
                appendValue(out, value, "####");
            }
        }
        if (!manifest.warnings().isEmpty()) {
            out.append("## 素材说明\n\n").append(String.join("\n\n", manifest.warnings())).append("\n");
        }
        return out.toString();
    }

    private void appendValue(StringBuilder out, Value value, String heading) {
        if (value.resource() == null) {
            if (value.text() != null && !value.text().isBlank()) {
                out.append(value.text()).append("\n\n");
            }
            if (value.mediaId() != null) {
                out.append("媒体：media://").append(value.mediaId()).append(" · 版本 ")
                        .append(value.mediaVersion()).append("\n\n");
            }
            return;
        }
        var data = value.data();
        out.append(heading).append(' ').append(kind(value.type())).append(" · ")
                .append(plain(data.path("title").asText(data.path("name").asText("#" + value.resource().id()))))
                .append("\n\n")
                .append("对象：").append(value.resource().type().code()).append('/').append(value.resource().id())
                .append(" · 版本 ").append(value.resource().version()).append("\n\n");
        data.fields().forEachRemaining(field -> {
            if (field.getValue().isNull() || field.getKey().startsWith("_") || List.of("id", "version", "title", "name",
                    "createTime", "updateTime", "deleted").contains(field.getKey())) {
                return;
            }
            String text = text(field.getValue());
            if (text.isBlank()) {
                return;
            }
            out.append("**").append(LABELS.getOrDefault(field.getKey(), field.getKey())).append("**\n\n").append(text)
                    .append("\n\n");
        });
    }

    private String nodeTitle(Manifest manifest, String nodeId) {
        return plain(
                manifest.graph().nodes().stream().filter(node -> node.id().equals(nodeId)).findFirst().orElseThrow()
                        .title());
    }

    private record ShotRow(Value value, String edge) {
    }

    public String csv(Manifest manifest) {
        var shots = new ArrayList<ShotRow>();
        manifest.values().stream().filter(value -> value.type() == CanvasValueType.SHOT)
                .forEach(value -> shots.add(new ShotRow(value, "")));
        manifest.graph().edges().forEach(edge -> {
            var value = manifest.fixedOutputs().get(edge.id());
            if (value != null && value.type() == CanvasValueType.SHOT) {
                shots.add(new ShotRow(value, edge.id()));
            }
        });
        boolean fixed = shots.stream().anyMatch(row -> !row.edge().isEmpty());
        var fields = new LinkedHashSet<String>(List.of("id", "version", "episodeId", "storyboardSceneId", "sortOrder"));
        shots.forEach(shot -> shot.value().data().fieldNames().forEachRemaining(fields::add));
        fields.removeIf(field -> field.startsWith("_"));
        StringBuilder out = new StringBuilder("\uFEFF");
        if (fixed) {
            out.append(csvCell("固定连线 [fixedEdgeId]")).append(',');
        }
        out.append(fields.stream().map(field -> csvCell(LABELS.getOrDefault(field, field) + " [" + field + "]"))
                .reduce((a, b) -> a + "," + b).orElse("")).append("\r\n");
        for (ShotRow row : shots) {
            var shot = row.value();
            if (fixed) {
                out.append(csvCell(row.edge())).append(',');
            }
            out.append(fields.stream().map(field -> csvCell(text(shot.data().path(field))))
                    .reduce((a, b) -> a + "," + b).orElse("")).append("\r\n");
        }
        return out.toString();
    }

    static String csvCell(String text) {
        String stripped = text.stripLeading();
        if (!stripped.isEmpty() && "=+-@".indexOf(stripped.charAt(0)) >= 0 || text.startsWith("\t") || text.startsWith(
                "\r")) {
            text = "'" + text;
        }
        return "\"" + text.replace("\"", "\"\"") + "\"";
    }

    private void zip(Manifest manifest, CanvasActor actor, Path target, Runnable check) throws IOException {
        try (var zip = new ZipOutputStream(Files.newOutputStream(target), StandardCharsets.UTF_8)) {
            entry(zip, "manifest.json", document(manifest));
            entry(zip, "script.md", markdown(manifest));
            entry(zip, "storyboard.csv", csv(manifest));
            entry(zip, "README.txt", "本包保存创建交付时的内容版本。manifest.json 包含对象版本、原始图与媒体路径映射。\n"
                    + "媒体文件名使用资源编号防止重名；原名、内容摘要、时长与来源见清单。CSV 对公式前缀做安全转义。\n"
                    + "此文件为交付快照；导入工作流请使用画布模板及显式资源绑定。\n");
            for (MediaResource file : manifest.media()) {
                check.run();
                Path plaintext = Files.createTempFile("afv-export-media-", ".part");
                try {
                    contents.copy(file.id(), actor, plaintext, MAX_BYTES, check);
                    if (!hash(plaintext).equals(file.content().contentHash())) {
                        throw new IOException("媒体文件摘要与交付清单不一致");
                    }
                    zip.putNextEntry(new ZipEntry(mediaPath(file)));
                    copy(plaintext, zip, check);
                    zip.closeEntry();
                } finally {
                    Files.deleteIfExists(plaintext);
                }
            }
        }
    }

    private void entry(ZipOutputStream zip, String name, String content) throws IOException {
        zip.putNextEntry(new ZipEntry(name));
        zip.write(content.getBytes(StandardCharsets.UTF_8));
        zip.closeEntry();
    }

    private void copy(Path file, OutputStream target, Runnable check) throws IOException {
        try (var input = Files.newInputStream(file)) {
            byte[] buffer = new byte[65536];
            long checkedAt = System.nanoTime();
            for (int length; (length = input.read(buffer)) >= 0; ) {
                target.write(buffer, 0, length);
                if (System.nanoTime() - checkedAt >= 3_000_000_000L) {
                    check.run();
                    checkedAt = System.nanoTime();
                }
            }
        }
    }

    public static String hash(Path file) throws IOException {
        try {
            var digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(file)) {
                byte[] buffer = new byte[65536];
                for (int length; (length = input.read(buffer)) >= 0; ) {
                    digest.update(buffer, 0, length);
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException failure) {
            throw new IllegalStateException("SHA-256 不可用", failure);
        }
    }

    public static String filename(Manifest manifest, Format format) {
        if (format == Format.MEDIA) {
            return safe(manifest.media().getFirst().name());
        }
        return safe(manifest.title()) + "-v" + manifest.documentRevision() + switch (format) {
            case MARKDOWN -> ".md";
            case CSV -> ".csv";
            case JSON -> ".json";
            case ZIP -> ".zip";
            case MEDIA -> throw new IllegalStateException("媒体文件已单独命名");
        };
    }

    public static String mime(Manifest manifest, Format format) {
        return switch (format) {
            case MARKDOWN -> "text/markdown; charset=UTF-8";
            case CSV -> "text/csv; charset=UTF-8";
            case JSON -> "application/json";
            case ZIP -> "application/zip";
            case MEDIA -> manifest.media().getFirst().content().mimeType();
        };
    }

    private static String safe(String value) {
        String name = value.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_").replaceAll("^\\.+|[ .]+$", "").strip();
        if (name.isEmpty()) {
            name = "交付文件";
        }
        return name.substring(0, Math.min(150, name.length()));
    }

    private static String mediaPath(MediaResource file) {
        return "media/" + file.id() + "-" + safe(file.name());
    }

    private String text(JsonNode node) {
        return node.isMissingNode() || node.isNull() ? "" : node.isTextual() ? node.asText() : node.toString();
    }

    private String plain(String text) {
        return text.replace('\n', ' ').replace('\r', ' ');
    }

    private String kind(CanvasValueType type) {
        return switch (type) {
            case PROJECT -> "项目";
            case SCRIPT -> "剧本";
            case SCRIPT_EPISODE -> "剧本分集";
            case SCRIPT_SCENE -> "剧本场次";
            case STORYBOARD -> "分镜";
            case STORYBOARD_EPISODE -> "分镜分集";
            case STORYBOARD_SCENE -> "分镜场次";
            case SHOT -> "镜头";
            case ASSET -> "资产";
            case ASSET_ITEM -> "资产变体";
            default -> type.code();
        };
    }
}
