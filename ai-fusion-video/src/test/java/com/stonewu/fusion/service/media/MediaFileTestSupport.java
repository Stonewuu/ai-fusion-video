package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/** 执行媒体工具并读取成片，用于验证画面时间戳和音轨内容。 */
public class MediaFileTestSupport {
    private final Path directory;
    private final ObjectMapper mapper = new ObjectMapper();

    public MediaFileTestSupport(Path directory) {
        this.directory = directory;
    }

    public String ffmpegPath() {
        return System.getProperty("test.ffmpeg.path", "ffmpeg");
    }

    public String ffprobePath() {
        return System.getProperty("test.ffprobe.path", "ffprobe");
    }

    public void ffmpeg(String... arguments) throws Exception {
        var command = new ArrayList<>(List.of(ffmpegPath(), "-v", "error", "-nostdin", "-y"));
        command.addAll(List.of(arguments));
        execute(command);
    }

    public JsonNode probe(Path input, String... arguments) throws Exception {
        var command = new ArrayList<>(List.of(ffprobePath(), "-v", "error"));
        command.addAll(List.of(arguments));
        command.addAll(List.of("-of", "json", input.toString()));
        return mapper.readTree(execute(command));
    }

    public int audioPeak(Path input, double startSeconds) throws Exception {
        Path samples = Files.createTempFile(directory, "samples-", ".pcm");
        try {
            ffmpeg("-ss", Double.toString(startSeconds), "-i", input.toString(), "-t", "0.2",
                    "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "s16le", samples.toString());
            var bytes = ByteBuffer.wrap(Files.readAllBytes(samples)).order(ByteOrder.LITTLE_ENDIAN);
            assertThat(bytes.remaining()).as("必须取得待检查时段的音频样本").isGreaterThan(0);
            int peak = 0;
            while (bytes.hasRemaining()) {
                peak = Math.max(peak, Math.abs((int) bytes.getShort()));
            }
            return peak;
        } finally {
            Files.deleteIfExists(samples);
        }
    }

    private String execute(List<String> command) throws Exception {
        Path log = Files.createTempFile(directory, "media-test-", ".log");
        Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        try {
            assertThat(process.waitFor(30, TimeUnit.SECONDS)).as("媒体工具执行超时：%s", command).isTrue();
            String output = Files.readString(log);
            assertThat(process.exitValue()).as("媒体工具执行失败：%s\n%s", command, output).isZero();
            return output;
        } finally {
            if (process.isAlive()) {
                process.destroyForcibly();
                process.waitFor(5, TimeUnit.SECONDS);
            }
            Files.deleteIfExists(log);
        }
    }
}
