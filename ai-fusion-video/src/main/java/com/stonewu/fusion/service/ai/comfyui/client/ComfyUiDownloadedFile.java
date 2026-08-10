package com.stonewu.fusion.service.ai.comfyui.client;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Temporary downloaded output that must be closed after platform persistence. */
public record ComfyUiDownloadedFile(
        Path path,
        String contentType,
        String extension,
        long size) implements AutoCloseable {

    @Override
    public void close() throws IOException {
        Files.deleteIfExists(path);
    }
}
