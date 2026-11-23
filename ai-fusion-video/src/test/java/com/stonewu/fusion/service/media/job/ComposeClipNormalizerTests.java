package com.stonewu.fusion.service.media.job;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.media.MediaFileTestSupport;
import com.stonewu.fusion.service.media.MediaProbe;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/** 通过生成文件的时长、帧时间戳与音频样本，保护分镜的画面边界和原声。 */
class ComposeClipNormalizerTests {
    @TempDir
    Path directory;
    private MediaFileTestSupport media;
    private MediaProbe probe;
    private ComposeClipNormalizer normalizer;

    @BeforeEach
    void setUp() {
        media = new MediaFileTestSupport(directory);
        probe = new MediaProbe(new ObjectMapper(), media.ffprobePath());
        normalizer = new ComposeClipNormalizer(probe, new MediaProcessRunner(), media.ffmpegPath());
    }

    @ParameterizedTest
    @CsvSource({"0,24", "1,25", "4,30", "15,30", "25,30", "25,30000/1001"})
    void keepsFourSecondsOfPictureAndAlignsOriginalOrSilentAudio(int audioSeconds, String frameRate)
            throws Exception {
        Path input = directory.resolve("input.mp4");
        var command = new ArrayList<>(List.of("-f", "lavfi", "-i",
                "testsrc2=size=160x90:rate=" + frameRate + ":duration=4"));
        if (audioSeconds > 0) {
            command.addAll(List.of("-f", "lavfi", "-i", "sine=sample_rate=48000:duration=" + audioSeconds));
        }
        command.addAll(List.of("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", input.toString()));
        media.ffmpeg(command.toArray(String[]::new));
        if (audioSeconds == 25) {
            assertThat(probe.inspect(input, "video/mp4").durationMs()).isBetween(24990L, 25030L);
            assertThat(probe.videoDurationSeconds(input).doubleValue()).isCloseTo(4, within(0.01));
        }

        Path output = directory.resolve("clip.mp4");
        var clip = normalizer.normalize(input, output, directory, 160, 90, audioSeconds > 0, () -> {});

        assertFourSecondClip(clip);
        if (audioSeconds == 0) {
            assertThat(media.audioPeak(output, 0.2)).isZero();
        } else {
            assertThat(media.audioPeak(output, 0.2)).isGreaterThan(200);
        }
        if (audioSeconds <= 1) {
            assertThat(media.audioPeak(output, 3.5)).isZero();
        } else {
            assertThat(media.audioPeak(output, 3.5)).isGreaterThan(200);
        }
        assertThat(Files.exists(directory.resolve("clip.mp4.picture.mp4"))).isFalse();
    }

    @Test
    void usesRenderedPictureForVariableFrameRateWebmWithoutStreamDuration() throws Exception {
        Path input = directory.resolve("input.webm");
        media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=4",
                "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=25",
                "-vf", "select='if(lt(t,2),1,not(mod(n,3))+eq(n,119))'", "-fps_mode", "vfr",
                "-c:v", "libvpx-vp9", "-deadline", "realtime", "-c:a", "libopus", input.toString());
        var original = media.probe(input, "-select_streams", "v:0", "-show_entries", "stream=duration");
        assertThat(original.path("streams").get(0).has("duration")).isFalse();

        var clip = normalizer.normalize(input, directory.resolve("clip.mp4"), directory, 160, 90, true, () -> {});

        assertFourSecondClip(clip);
        assertThat(media.audioPeak(clip.path(), 3.5)).isGreaterThan(200);
    }

    @Test
    void preservesDelayedAudioInsteadOfMovingSpeechToTheFirstFrame() throws Exception {
        Path input = directory.resolve("delayed.mp4");
        media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=4",
                "-itsoffset", "1", "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", input.toString());

        var clip = normalizer.normalize(input, directory.resolve("clip.mp4"), directory, 160, 90, true, () -> {});

        assertFourSecondClip(clip);
        assertThat(media.audioPeak(clip.path(), 0.2)).isZero();
        assertThat(media.audioPeak(clip.path(), 1.2)).isGreaterThan(200);
        assertThat(media.audioPeak(clip.path(), 3)).isZero();
    }

    private void assertFourSecondClip(ComposeClipNormalizer.Clip clip) throws Exception {
        assertThat(clip.durationSeconds().doubleValue()).isCloseTo(4, within(0.000001));
        var actual = media.probe(clip.path(), "-show_entries",
                "format=duration:stream=codec_type,duration,sample_rate,channels");
        assertThat(actual.path("format").path("duration").asDouble()).isCloseTo(4, within(0.002));
        assertThat(actual.path("streams").size()).isEqualTo(2);
        for (var stream : actual.path("streams")) {
            assertThat(stream.path("duration").asDouble()).isCloseTo(4, within(0.002));
            if (stream.path("codec_type").asText().equals("audio")) {
                assertThat(stream.path("sample_rate").asInt()).isEqualTo(48000);
                assertThat(stream.path("channels").asInt()).isEqualTo(2);
            }
        }
        var frames = media.probe(clip.path(), "-select_streams", "v:0", "-show_frames", "-show_entries",
                "frame=best_effort_timestamp_time").path("frames");
        assertThat(frames.size()).isEqualTo(120);
        for (int index = 1; index < frames.size(); index++) {
            double gap = frames.get(index).path("best_effort_timestamp_time").asDouble()
                    - frames.get(index - 1).path("best_effort_timestamp_time").asDouble();
            assertThat(gap).as("相邻画面帧必须连续").isCloseTo(1.0 / 30, within(0.000002));
        }
    }
}
