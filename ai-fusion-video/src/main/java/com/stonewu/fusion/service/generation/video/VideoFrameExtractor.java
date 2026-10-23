package com.stonewu.fusion.service.generation.video;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * 纯文件处理，不下载地址、不决定所有者，也不把帧写入公开目录。
 */
@Component
public class VideoFrameExtractor {
    @Value("${video.compose.ffmpeg-path:ffmpeg}")
    private String ffmpegPath;
    @Value("${video.frame-extraction.timeout-seconds:60}")
    private long timeoutSeconds;

    public ExtractedFrames extract(Path input, Path outputDirectory, boolean first, boolean last) throws IOException {
        if (!Files.isRegularFile(input) || !Files.isDirectory(outputDirectory)) {
            throw new IOException("视频提帧输入或目录不可用");
        }
        Path firstFrame = first ? extract(input, outputDirectory.resolve("first.jpg"), false) : null;
        Path lastFrame = last ? extract(input, outputDirectory.resolve("last.jpg"), true) : null;
        return new ExtractedFrames(firstFrame, lastFrame);
    }

    private Path extract(Path input, Path output, boolean last) throws IOException {
        if (ffmpegPath == null || ffmpegPath.isBlank() || timeoutSeconds <= 0) {
            throw new IOException("视频提帧配置无效");
        }
        List<String> command = new ArrayList<>(List.of(ffmpegPath, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-y", "-protocol_whitelist", "file,pipe", "-i", input.toAbsolutePath().toString(),
                "-map", "0:v:0", "-an", "-sn", "-dn", "-q:v", "2"));
        if (last) {
            // 顺序解码并覆盖同一张输出，得到实际最后一帧；不靠时长或帧率猜测尾帧位置。
            command.addAll(List.of("-fps_mode", "passthrough", "-update", "1"));
        } else {
            command.addAll(List.of("-frames:v", "1"));
        }
        command.add(output.toAbsolutePath().toString());
        Process process = new ProcessBuilder(command).redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .redirectError(ProcessBuilder.Redirect.DISCARD).start();
        try {
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(5, TimeUnit.SECONDS);
                throw new IOException("视频提帧超时");
            }
        } catch (InterruptedException interrupted) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("视频提帧被中断", interrupted);
        }
        if (process.exitValue() != 0 || !Files.isRegularFile(output) || Files.size(output) == 0) {
            throw new IOException((last ? "尾帧" : "首帧") + "提取失败，FFmpeg 退出码：" + process.exitValue());
        }
        return output;
    }

    public record ExtractedFrames(Path firstFrame, Path lastFrame) {
    }
}
