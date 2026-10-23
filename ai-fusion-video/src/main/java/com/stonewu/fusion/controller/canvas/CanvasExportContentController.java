package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.service.canvas.delivery.CanvasExportRepository;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportTickets;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaByteRange;
import com.stonewu.fusion.service.media.model.MediaException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;

@RestController
@RequiredArgsConstructor
public class CanvasExportContentController {
    private final CanvasExportTickets tickets;
    private final CanvasExportRepository jobs;
    private final MediaBlobStore blobs;

    @RequestMapping(value = "/api/canvas-export-content/{id}", method = {RequestMethod.GET, RequestMethod.HEAD})
    public void content(@PathVariable String id, @RequestParam String ticket, HttpServletRequest request,
                        HttpServletResponse response) throws IOException {
        var grant = tickets.authorize(id, ticket);
        var job = grant.job();
        long size = job.byteSize();
        String etag = "\"" + job.contentHash() + "\"";
        String rangeHeader = request.getHeader(HttpHeaders.RANGE);
        if (request.getHeader(HttpHeaders.IF_RANGE) != null && !etag.equals(request.getHeader(HttpHeaders.IF_RANGE))) {
            rangeHeader = null;
        }
        MediaByteRange range;
        try {
            range = MediaByteRange.parse(rangeHeader, size);
        } catch (MediaException invalid) {
            response.setHeader(HttpHeaders.CONTENT_RANGE, "bytes */" + size);
            throw invalid;
        }
        try (var input = blobs.open(jobs.blob(job.blobId()))) {
            if (input.size() != size) {
                throw new IOException("交付文件内容长度不符");
            }
            input.position(range.start());
            response.setStatus(range.partial() ? 206 : 200);
            response.setContentType(job.mimeType());
            response.setContentLengthLong(range.length());
            response.setHeader(HttpHeaders.ACCEPT_RANGES, "bytes");
            response.setHeader(HttpHeaders.ETAG, etag);
            response.setHeader(HttpHeaders.CACHE_CONTROL, "private, no-store");
            response.setHeader("X-Content-Type-Options", "nosniff");
            response.setHeader("Referrer-Policy", "no-referrer");
            response.setHeader("Content-Security-Policy", "sandbox");
            response.setHeader(HttpHeaders.CONTENT_DISPOSITION,
                    ContentDisposition.attachment().filename(job.filename(), StandardCharsets.UTF_8).build()
                            .toString());
            if (range.partial()) {
                response.setHeader(HttpHeaders.CONTENT_RANGE,
                        "bytes " + range.start() + "-" + range.end() + "/" + size);
            }
            if (request.getMethod().equals("HEAD")) {
                return;
            }
            jobs.event(id, grant.actor().userId(), "DOWNLOAD_STARTED",
                    Map.of("start", range.start(), "length", range.length()));
            long remaining = range.length();
            long checkedAt = System.nanoTime();
            long checkedBytes = 0;
            var buffer = ByteBuffer.allocate(65536);
            int empty = 0;
            while (remaining > 0) {
                if (checkedBytes >= 8 * 1024 * 1024 || System.nanoTime() - checkedAt >= 5_000_000_000L) {
                    tickets.authorize(id, ticket);
                    checkedAt = System.nanoTime();
                    checkedBytes = 0;
                }
                buffer.clear().limit((int) Math.min(buffer.capacity(), remaining));
                int read = input.read(buffer);
                if (read < 0) {
                    throw new IOException("交付文件内容不完整");
                }
                if (read == 0) {
                    if (++empty > 8) {
                        throw new IOException("交付文件读取无进展");
                    }
                    continue;
                }
                empty = 0;
                response.getOutputStream().write(buffer.array(), 0, read);
                remaining -= read;
                checkedBytes += read;
            }
        }
    }
}
