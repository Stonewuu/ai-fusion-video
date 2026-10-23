package com.stonewu.fusion.service.storyboard;

import com.stonewu.fusion.config.EpisodeBgmProperties;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/** 显式启用真实 ffmpeg 回归：-Dtest.ffmpeg=true。 */
@EnabledIfSystemProperty(named = "test.ffmpeg", matches = "true")
class EpisodeBgmMediaTests {
    @TempDir
    Path dir;

    @ParameterizedTest
    @CsvSource({"1, 4", "4, 1", "1, 1", "4, 8"})
    void preservesVideoDurationRegardlessOfAudioDurations(double originalSeconds, double musicSeconds)
            throws Exception {
        Path video = dir.resolve("original.mp4");
        Path music = dir.resolve("music.m4a");
        Path output = dir.resolve("scored.mp4");
        run(List.of("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=160x120:r=25:d=4", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=" + originalSeconds, "-c:v", "libx264", "-pix_fmt",
                "yuv420p", "-c:a", "aac", video.toString()));
        run(List.of("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "sine=frequency=880:duration=" + musicSeconds, "-c:a", "aac", music.toString()));
        EpisodeBgmService service = new EpisodeBgmService(new EpisodeBgmProperties(), null);
        run(service.buildMixCommand(video, music, null, true, 4, output));
        Process probe = new ProcessBuilder("ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration", "-of", "csv=p=0", output.toString()).start();
        assertThat(probe.waitFor(10, TimeUnit.SECONDS)).isTrue();
        assertThat(probe.exitValue()).isZero();
        double duration = Double.parseDouble(new String(probe.getInputStream().readAllBytes()).trim());
        assertThat(duration).isCloseTo(4, within(0.05));
    }

    private void run(List<String> command) throws Exception {
        Path log = Files.createTempFile(dir, "ffmpeg-", ".log");
        Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        boolean done = process.waitFor(30, TimeUnit.SECONDS);
        if (!done) process.destroyForcibly();
        assertThat(done).as("ffmpeg 超时: %s", command).isTrue();
        assertThat(process.exitValue()).as(Files.readString(log)).isZero();
    }
}
