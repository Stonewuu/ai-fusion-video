package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.service.media.MediaIntakeService;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

@Component
public class MediaProcessRunner {
    public void run(List<String> command, Path work, Path output, Runnable active) throws IOException {
        Path log = Files.createTempFile(work, "process-", ".log");
        Process process = null;
        long until = System.nanoTime() + TimeUnit.MINUTES.toNanos(30);
        try {
            active.run();
            try {
                process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(log.toFile()).start();
            } catch (IOException failure) {
                throw new IOException("无法启动 FFmpeg，请检查 video.compose.ffmpeg-path", failure);
            }
            process.getOutputStream().close();
            while (!process.waitFor(1, TimeUnit.SECONDS)) {
                active.run();
                if (System.nanoTime() > until) {
                    throw new IOException("媒体处理超过 30 分钟时限");
                }
                if (Files.exists(output) && Files.size(output) > MediaIntakeService.MAX_RESULT_BYTES) {
                    throw new IOException("媒体处理结果超过 2 GB 限制");
                }
                if (Files.size(log) > 1024 * 1024) {
                    throw new IOException("媒体处理输出异常，请检查输入格式");
                }
            }
            active.run();
            if (process.exitValue() != 0) {
                String error = Files.readString(log);
                throw new IOException("FFmpeg 处理失败：" + error.substring(0, Math.min(error.length(), 1500)));
            }
            if (!Files.exists(output) || Files.size(output) == 0 || Files.size(
                    output) > MediaIntakeService.MAX_RESULT_BYTES) {
                throw new IOException("媒体处理结果为空或超过大小限制");
            }
        } catch (InterruptedException cancelled) {
            Thread.currentThread().interrupt();
            throw new IOException("媒体处理已停止", cancelled);
        } finally {
            if (process != null && process.isAlive()) {
                process.descendants().forEach(ProcessHandle::destroyForcibly);
                process.destroyForcibly();
                try {
                    process.waitFor(5, TimeUnit.SECONDS);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
            Files.deleteIfExists(log);
        }
    }
}
