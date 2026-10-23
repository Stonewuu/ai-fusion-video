package com.stonewu.fusion.service.canvas.run.media;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.MediaContentReader;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobOutputs;
import com.stonewu.fusion.service.media.job.MediaProcessRunner;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasLocalMediaProcessor {
    private final CanvasMediaJobs tasks;
    private final MediaJobOutputs outputs;
    private final MediaContentReader content;
    private final MediaResourceQueries resources;
    private final MediaProcessRunner processes;
    private final CanvasRunSources sources;
    private final CanvasJson json;
    private final Environment environment;

    public Ports process(MediaJob job, CanvasMediaInput input, Path work) throws IOException {
        input.inputs().all().forEach(value -> sources.authorize(value, job.context().actor(), true));
        CanvasMediaRules.inputs(input.operation(), input.inputs());
        return switch (input.operation()) {
            case IMAGE_TRANSFORM -> transform(job, input, work);
            case FRAME_EXTRACT -> frames(job, input, work);
            case VIDEO_COMPOSE -> compose(job, input, work);
            case AUDIO_MIX -> mix(job, input, work);
            default -> throw CanvasRunException.invalid("不是本地媒体处理能力");
        };
    }

    private Ports transform(MediaJob job, CanvasMediaInput input, Path work) throws IOException {
        var existing = existing(job, "image", MediaKind.IMAGE);
        if (existing != null) {
            return new Ports(Map.of("image", List.of(existing)));
        }
        var media = input.inputs().at("image").getFirst();
        var metadata = resources.get(media.mediaId(), job.context().actor()).content();
        var config = input.config();
        int width = config.path("width").asInt(), height = config.path("height").asInt();
        String filter;
        if (config.path("operation").asText().equals("crop")) {
            int x = config.path("x").asInt(0), y = config.path("y").asInt(0);
            if (metadata.width() == null || metadata.height() == null || (long) x + width > metadata.width() || (long) y + height > metadata.height()) {
                throw CanvasRunException.invalid("裁剪范围超出原图尺寸");
            }
            filter = "crop=" + width + ":" + height + ":" + x + ":" + y;
        } else {
            filter = switch (config.path("fit").asText("contain")) {
                case "fill" -> "scale=" + width + ":" + height;
                case "cover" ->
                        "scale=" + width + ":" + height + ":force_original_aspect_ratio=increase,crop=" + width + ":" + height;
                case "contain" ->
                        "scale=" + width + ":" + height + ":force_original_aspect_ratio=decrease,pad=" + width + ":" + height + ":(ow-iw)/2:(oh-ih)/2:color=black@0";
                default -> throw CanvasRunException.invalid("图片缩放方式未登记");
            };
        }
        Path original = copy(job, media, work.resolve("source.image"));
        Path result = work.resolve("image.png");
        var command = base(original);
        command.addAll(List.of("-vf", filter, "-frames:v", "1", "-update", "1", result.toString()));
        processes.run(command, work, result, () -> tasks.check(job.claim(), false));
        return new Ports(Map.of("image", List.of(store(job, "image", result, "image/png", MediaKind.IMAGE))));
    }

    private Ports frames(MediaJob job, CanvasMediaInput input, Path work) throws IOException {
        Map<String, List<Value>> result = new LinkedHashMap<>();
        String mode = input.config().path("frameType").asText();
        Path original = null;
        for (String port : mode.equals("both") ? List.of("firstFrame", "lastFrame") : List.of(
                mode.equals("first") ? "firstFrame" : "lastFrame")) {
            var previous = existing(job, port, MediaKind.IMAGE);
            if (previous != null) {
                result.put(port, List.of(previous));
                continue;
            }
            if (original == null) {
                original = copy(job, input.inputs().at("video").getFirst(), work.resolve("source.video"));
            }
            var command = new ArrayList<>(
                    List.of(ffmpeg(), "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe"));
            if (port.equals("lastFrame")) {
                command.addAll(List.of("-sseof", "-1"));
            }
            command.addAll(List.of("-i", original.toString(), "-an"));
            if (port.equals("firstFrame")) {
                command.addAll(List.of("-frames:v", "1"));
            }
            Path image = work.resolve(port + ".png");
            command.addAll(List.of("-fps_mode", "passthrough", "-update", "1", image.toString()));
            processes.run(command, work, image, () -> tasks.check(job.claim(), false));
            result.put(port, List.of(store(job, port, image, "image/png", MediaKind.IMAGE)));
        }
        return new Ports(result);
    }

    private Ports compose(MediaJob job, CanvasMediaInput input, Path work) throws IOException {
        var previous = existing(job, "video", MediaKind.VIDEO);
        if (previous != null) {
            return new Ports(Map.of("video", List.of(previous)));
        }
        List<Value> clips = input.inputs().at("videos");
        if (!input.inputs().at("episode").isEmpty()) {
            var frozen = input.inputs().at("episode").getFirst().data().path("_canvasVideos");
            if (!frozen.isArray() || frozen.isEmpty()) {
                throw CanvasRunException.invalid("此分镜集缺少已登记的视频快照，请检查每个镜头的视频");
            }
            clips = new ArrayList<>();
            for (var value : frozen) {
                clips.add(json.read(value.toString(), Value.class));
            }
        }
        if (clips.isEmpty() || clips.size() > 1000) {
            throw CanvasRunException.invalid("合成视频数量必须为 1 至 1000");
        }
        var first = resources.get(clips.getFirst().mediaId(), job.context().actor()).content();
        int width, height;
        if (input.config().has("resolution")) {
            String[] parts = input.config().path("resolution").asText().split("x");
            width = Integer.parseInt(parts[0]);
            height = Integer.parseInt(parts[1]);
        } else {
            if (first.width() == null || first.height() == null) {
                throw CanvasRunException.invalid("第一个视频缺少画面尺寸");
            }
            width = first.width() / 2 * 2;
            height = first.height() / 2 * 2;
        }
        if (width < 2 || height < 2 || width > 4096 || height > 4096 || width % 2 != 0 || height % 2 != 0) {
            throw CanvasRunException.invalid("合成宽高必须为 2 至 4096 之间的偶数");
        }
        int fps = input.config().path("frameRate").asInt(30);
        long duration = 0;
        var normalized = new ArrayList<Path>();
        for (int index = 0; index < clips.size(); index++) {
            var clip = clips.get(index);
            var facts = resources.get(clip.mediaId(), job.context().actor()).content();
            if (facts.durationMs() == null || (duration += facts.durationMs()) > 3_600_000L) {
                throw CanvasRunException.invalid("本次合成总时长不能超过一小时");
            }
            Path source = copy(job, clip, work.resolve("input-" + index + ".video"));
            Path output = work.resolve("clip-" + index + ".mp4");
            var command = base(source);
            if (!facts.hasAudio()) {
                command.addAll(List.of("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"));
            }
            command.addAll(List.of("-map", "0:v:0", "-map", facts.hasAudio() ? "0:a:0" : "1:a:0", "-vf",
                    "scale=" + width + ":" + height + ":force_original_aspect_ratio=decrease,pad=" + width + ":" + height + ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" + fps,
                    "-af", "apad", "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt",
                    "yuv420p",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2", "-video_track_timescale", "90000", output.toString()));
            processes.run(command, work, output, () -> tasks.check(job.claim(), false));
            normalized.add(output);
            Files.delete(source);
        }
        Path manifest = work.resolve("concat.txt");
        Files.writeString(manifest,
                String.join("\n", normalized.stream().map(path -> "file '" + path.getFileName() + "'").toList()),
                StandardCharsets.UTF_8);
        Path output = work.resolve("composed.mp4");
        processes.run(
                List.of(ffmpeg(), "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-f", "concat",
                        "-safe", "1",
                        "-i", manifest.toString(), "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-movflags",
                        "+faststart", output.toString()),
                work, output, () -> tasks.check(job.claim(), false));
        return new Ports(Map.of("video", List.of(store(job, "video", output, "video/mp4", MediaKind.VIDEO))));
    }

    private Ports mix(MediaJob job, CanvasMediaInput input, Path work) throws IOException {
        var previous = existing(job, "video", MediaKind.VIDEO);
        if (previous != null) {
            return new Ports(Map.of("video", List.of(previous)));
        }
        var video = input.inputs().at("video").getFirst();
        var facts = resources.get(video.mediaId(), job.context().actor()).content();
        if (facts.durationMs() == null || facts.durationMs() <= 0 || facts.durationMs() > 3_600_000L) {
            throw CanvasRunException.invalid("混音视频时长必须在一小时以内");
        }
        var command = base(copy(job, video, work.resolve("source.video")));
        List<String> filters = new ArrayList<>();
        List<String> tracks = new ArrayList<>();
        if (facts.hasAudio()) {
            filters.add("[0:a:0]volume=" + input.config().path("originalVolume").asDouble(1) + ",apad[a0]");
            tracks.add("[a0]");
        }
        int index = 1;
        for (String port : List.of("music", "sfx")) {
            if (input.inputs().at(port).isEmpty()) {
                continue;
            }
            Path audio = copy(job, input.inputs().at(port).getFirst(), work.resolve(port + ".audio"));
            if (port.equals("music")) {
                command.addAll(List.of("-stream_loop", "-1"));
            }
            command.addAll(List.of("-protocol_whitelist", "file,pipe", "-i", audio.toString()));
            filters.add("[" + index + ":a:0]volume=" + input.config().path(port + "Volume")
                    .asDouble(port.equals("music") ? 0.35 : 1) + ",apad[a" + index + "]");
            tracks.add("[a" + index + "]");
            index++;
        }
        command.addAll(List.of("-map", "0:v:0", "-c:v", "copy"));
        if (!tracks.isEmpty()) {
            filters.add(
                    String.join("", tracks) + "amix=inputs=" + tracks.size() + ":normalize=0:duration=longest[mix]");
            command.addAll(List.of("-filter_complex", String.join(";", filters), "-map", "[mix]", "-c:a", "aac", "-ar",
                    "48000", "-ac", "2"));
        } else {
            command.add("-an");
        }
        Path output = work.resolve("mixed.mp4");
        command.addAll(List.of("-t", String.valueOf(facts.durationMs() / 1000.0), "-movflags", "+faststart",
                output.toString()));
        processes.run(command, work, output, () -> tasks.check(job.claim(), false));
        return new Ports(Map.of("video", List.of(store(job, "video", output, "video/mp4", MediaKind.VIDEO))));
    }

    private ArrayList<String> base(Path source) {
        return new ArrayList<>(
                List.of(ffmpeg(), "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-i",
                        source.toString()));
    }

    private String ffmpeg() {
        return environment.getProperty("video.compose.ffmpeg-path", "ffmpeg");
    }

    private Path copy(MediaJob job, Value value, Path target) throws IOException {
        tasks.check(job.claim(), false);
        sources.authorize(value, job.context().actor(), true);
        content.copy(value.mediaId(), job.context().actor(), target, MediaIntakeService.MAX_RESULT_BYTES);
        sources.authorize(value, job.context().actor(), true);
        return target;
    }

    private Value existing(MediaJob job, String slot, MediaKind kind) {
        return outputs.existing(job.claim(), slot, kind)
                .map(reference -> sources.mediaValue(MediaReference.id(reference), job.context().actor(),
                        job.projectId())).orElse(null);
    }

    private Value store(MediaJob job, String slot, Path file, String mime, MediaKind kind) throws IOException {
        return sources.mediaValue(MediaReference.id(outputs.store(job.claim(), slot, file, mime, kind)),
                job.context().actor(), job.projectId());
    }
}
