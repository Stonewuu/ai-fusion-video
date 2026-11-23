package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.service.media.MediaIoBoundary;
import com.stonewu.fusion.service.media.MediaProbe;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * 先确定规范化画面的时间范围，再将原声或静音对齐到同一范围。画面只编码一次。
 */
@Component
public class ComposeClipNormalizer {
    private final MediaProbe probe;
    private final MediaProcessRunner processes;
    private final String ffmpeg;

    public ComposeClipNormalizer(MediaProbe probe, MediaProcessRunner processes,
                                 @Value("${video.compose.ffmpeg-path:ffmpeg}") String ffmpeg) {
        this.probe = probe;
        this.processes = processes;
        this.ffmpeg = ffmpeg;
    }

    public Clip normalize(Path input, Path output, Path work, int width, int height, boolean audio,
                          Runnable active) throws IOException {
        MediaIoBoundary.requireOutsideTransaction();
        Path picture = work.resolve(output.getFileName() + ".picture.mp4");
        try {
            processes.run(pictureCommand(input, picture, width, height), work, picture, active);
            BigDecimal duration = probe.videoDurationSeconds(picture);
            processes.run(audioCommand(input, picture, output, audio, duration), work, output, active);
            return new Clip(output, duration);
        } finally {
            Files.deleteIfExists(picture);
        }
    }

    private List<String> pictureCommand(Path input, Path output, int width, int height) {
        // 两路均以 FFmpeg 读取后的输入时间轴为准，保留原声相对画面的起始偏移。
        return List.of(ffmpeg, "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe",
                "-i", input.toString(), "-map", "0:v:0", "-an", "-vf",
                "scale=" + width + ":" + height + ":force_original_aspect_ratio=decrease,pad="
                        + width + ":" + height + ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30:start_time=0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-video_track_timescale", "90000", output.toString());
    }

    private List<String> audioCommand(Path input, Path picture, Path output, boolean audio, BigDecimal duration) {
        List<String> command = new ArrayList<>(List.of(ffmpeg, "-v", "error", "-nostdin", "-y",
                "-protocol_whitelist", "file,pipe", "-i", picture.toString()));
        if (audio) {
            command.addAll(List.of("-protocol_whitelist", "file,pipe", "-i", input.toString()));
        } else {
            command.addAll(List.of("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"));
        }
        String seconds = duration.toPlainString();
        command.addAll(List.of("-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af",
                "aresample=48000:async=1:first_pts=0,apad=whole_dur=" + seconds + ",atrim=duration=" + seconds,
                "-t", seconds, "-c:a", "aac", "-ar", "48000", "-ac", "2",
                "-video_track_timescale", "90000", output.toString()));
        return command;
    }

    public record Clip(Path path, BigDecimal durationSeconds) {
    }
}
