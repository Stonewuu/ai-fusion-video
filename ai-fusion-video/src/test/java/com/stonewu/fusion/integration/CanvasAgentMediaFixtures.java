package com.stonewu.fusion.integration;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.imageio.ImageIO;
import static org.assertj.core.api.Assertions.assertThat;

final class CanvasAgentMediaFixtures {
    private CanvasAgentMediaFixtures() { }
    record File(String kind, String mime, String port, byte[] bytes) { }

    static List<File> create(Path directory) throws Exception {
        Files.createDirectories(directory);
        var image = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", image);
        Path video = directory.resolve("reference.mp4"), audio = directory.resolve("reference.wav");
        render(directory, "video", List.of("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=2", "-t", "1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", video.toString()));
        render(directory, "audio", List.of("-f", "lavfi", "-i", "sine=frequency=320:duration=0.2",
                "-c:a", "pcm_s16le", audio.toString()));
        return List.of(new File("image", "image/png", "referenceImages", image.toByteArray()),
                new File("video", "video/mp4", "referenceVideos", Files.readAllBytes(video)),
                new File("audio", "audio/wav", "referenceAudios", Files.readAllBytes(audio)),
                new File("document", "application/pdf", "referenceDocuments", document()));
    }

    static byte[] video(Path directory, int seconds) throws Exception {
        Files.createDirectories(directory);
        Path output = directory.resolve("generated.mp4");
        render(directory, "generated", List.of("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=4", "-t", String.valueOf(seconds),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", output.toString()));
        return Files.readAllBytes(output);
    }

    private static void render(Path directory, String kind, List<String> args) throws Exception {
        var command = new ArrayList<>(List.of("ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"));
        command.addAll(args);
        var process = new ProcessBuilder(command).redirectError(directory.resolve(kind + ".log").toFile())
                .redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
        if (!process.waitFor(20, TimeUnit.SECONDS)) { process.destroyForcibly(); throw new IllegalStateException("测试媒体生成超时"); }
        assertThat(process.exitValue()).as(kind + " fixture").isZero();
    }

    private static byte[] document() throws Exception {
        String content = "BT /F1 12 Tf 24 100 Td (Canvas mixed media test.) Tj ET\n";
        var objects = List.of("<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 150] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                "<< /Length " + content.length() + " >>\nstream\n" + content + "endstream");
        var output = new ByteArrayOutputStream();
        output.write("%PDF-1.4\n".getBytes(StandardCharsets.US_ASCII));
        List<Integer> offsets = new ArrayList<>();
        for (String object : objects) {
            offsets.add(output.size());
            output.write((offsets.size() + " 0 obj\n" + object + "\nendobj\n").getBytes(StandardCharsets.US_ASCII));
        }
        int xref = output.size();
        output.write(("xref\n0 6\n0000000000 65535 f \n").getBytes(StandardCharsets.US_ASCII));
        for (int offset : offsets) output.write(String.format("%010d 00000 n \n", offset).getBytes(StandardCharsets.US_ASCII));
        output.write(("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + xref + "\n%%EOF\n").getBytes(StandardCharsets.US_ASCII));
        return output.toByteArray();
    }
}
