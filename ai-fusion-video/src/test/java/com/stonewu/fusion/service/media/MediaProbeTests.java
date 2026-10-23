package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.media.model.MediaKind;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import static org.assertj.core.api.Assertions.assertThat;

class MediaProbeTests {
    @TempDir Path directory;

    @Test
    void distinguishesWebmMatroskaAndQuickTimeByContainerInsteadOfFileExtension() throws Exception {
        var probe = new MediaProbe(new ObjectMapper(), "ffprobe");
        String[][] cases = {{"webm", "libvpx-vp9", "video/webm"}, {"matroska", "libx264", "video/x-matroska"}, {"mov", "libx264", "video/quicktime"}};
        for (String[] item : cases) {
            Path output = directory.resolve(item[0] + ".unknown");
            Process process = new ProcessBuilder("ffmpeg", "-y", "-f", "lavfi", "-i",
                    "color=c=red:s=64x48:r=6:d=0.5", "-an", "-c:v", item[1], "-f", item[0], output.toString())
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD).redirectError(ProcessBuilder.Redirect.DISCARD).start();
            try {
                assertThat(process.waitFor(30, TimeUnit.SECONDS)).isTrue();
                assertThat(process.exitValue()).isZero();
            } finally { if (process.isAlive()) process.destroyForcibly(); }
            var facts = probe.inspect(output, "application/octet-stream");
            assertThat(facts.kind()).isEqualTo(MediaKind.VIDEO);
            assertThat(facts.mimeType()).isEqualTo(item[2]);
            assertThat(facts.durationMs()).isBetween(450L, 750L);
            assertThat(facts.width()).isEqualTo(64);
        }
    }
}
