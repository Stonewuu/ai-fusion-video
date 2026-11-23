package com.stonewu.fusion.service.media.job;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.MediaContentReader;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaJobOrigin;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * 每个镜头先统一画面和音轨，再无损连接规范化片段；无音轨镜头补静音而不丢弃其他镜头的原声。
 */
@Component
@RequiredArgsConstructor
public class ComposeVideoProcessor {
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final MediaJobOutputs outputs;
    private final MediaSourceImportService sources;
    private final MediaResourceQueries resources;
    private final MediaContentReader content;
    private final MediaProcessRunner processes;
    private final ComposeClipNormalizer clips;
    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpeg = "ffmpeg";

    public MediaJob prepare(MediaJob.Claim claim, Path work) throws IOException {
        var job = jobs.check(claim);
        var checkpoint = job.checkpoint().deepCopy();
        var input = json.read(json.write(job.input()), ComposeSnapshot.class);
        ArrayNode ids = checkpoint.has("inputs") ? (ArrayNode) checkpoint.get("inputs") : checkpoint.putArray("inputs");
        for (int index = ids.size(); index < input.clips().size(); index++) {
            jobs.check(claim);
            var media = sources.importJobSource(input.clips().get(index), job.context(), new MediaJobOrigin(claim),
                    () -> jobs.check(claim));
            if (media.content().kind() != MediaKind.VIDEO) {
                throw MediaException.invalid("合成输入必须是视频");
            }
            ids.add(media.id());
            jobs.checkpoint(claim, checkpoint, "准备镜头 " + (index + 1) + "/" + input.clips().size());
        }
        if (checkpoint.hasNonNull("baseVideo")) {
            return jobs.check(claim);
        }
        var previous = outputs.existing(claim, "base", MediaKind.VIDEO);
        if (previous.isPresent()) {
            checkpoint.put("baseVideo", previous.get());
            return jobs.checkpoint(claim, checkpoint, "已恢复合成视频");
        }
        var first = resources.get(ids.get(0).asText(), job.context().actor()).content();
        if (first.width() == null || first.height() == null || first.width() < 2 || first.height() < 2
                || first.width() > 4096 || first.height() > 4096) {
            throw MediaException.invalid("合成画面尺寸必须在 2 至 4096 像素之间");
        }
        int width = first.width() / 2 * 2;
        int height = first.height() / 2 * 2;
        List<ComposeClipNormalizer.Clip> normalized = new ArrayList<>();
        for (int index = 0; index < ids.size(); index++) {
            var media = resources.get(ids.get(index).asText(), job.context().actor());
            Path source = work.resolve("input-" + index + ".media");
            content.copy(media.id(), job.context().actor(), source, MediaIntakeService.MAX_RESULT_BYTES);
            Path target = work.resolve("clip-" + index + ".mp4");
            jobs.checkpoint(claim, checkpoint, "合成镜头 " + (index + 1) + "/" + ids.size());
            normalized.add(clips.normalize(source, target, work, width, height, media.content().hasAudio(),
                    () -> jobs.check(claim)));
            Files.delete(source);
        }
        Path list = work.resolve("concat.txt");
        // 文件名由服务端生成；下一镜头按画面时长开始，避免音频编码尾部影响镜头边界。
        Files.writeString(list,
                String.join("\n", normalized.stream().map(clip -> "file '" + clip.path().getFileName()
                        + "'\nduration " + clip.durationSeconds().toPlainString()).toList()),
                StandardCharsets.UTF_8);
        Path result = work.resolve("base.mp4");
        processes.run(
                List.of(ffmpeg, "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-f", "concat",
                        "-safe", "1",
                        "-i", list.toString(), "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-movflags",
                        "+faststart", result.toString()),
                work, result, () -> jobs.check(claim));
        checkpoint.put("baseVideo", outputs.store(claim, "base", result, "video/mp4", MediaKind.VIDEO));
        return jobs.checkpoint(claim, checkpoint, "镜头合成完成");
    }

}
