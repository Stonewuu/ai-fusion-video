package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.MediaAccessTickets;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaByteRange;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.web.bind.annotation.*;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;

/**
 * 此入口不使用登录 Cookie；每次都验证短时凭据及其持有者的当前权限。
 */
@RestController
@RequiredArgsConstructor
public class MediaContentController {
    private final MediaAccessTickets tickets;
    private final MediaResourceRepository resources;
    private final MediaBlobStore blobs;

    @RequestMapping(value = "/api/media-content/{id}", method = {RequestMethod.GET, RequestMethod.HEAD})
    public void content(@PathVariable String id, @RequestParam String ticket,
                        HttpServletRequest request, HttpServletResponse response) throws IOException {
        var grant = tickets.authorize(id, ticket);
        var media = grant.resource();
        long size = media.content().byteSize();
        String etag = "\"" + id + "-" + media.content().contentHash() + "\"";
        String rangeHeader = request.getHeader(HttpHeaders.RANGE);
        String ifRange = request.getHeader(HttpHeaders.IF_RANGE);
        if (ifRange != null && !ifRange.equals(etag)) {
            rangeHeader = null;
        }
        MediaByteRange range;
        try {
            range = MediaByteRange.parse(rangeHeader, size);
        } catch (MediaException invalid) {
            response.setHeader(HttpHeaders.CONTENT_RANGE, "bytes */" + size);
            throw invalid;
        }
        try (var input = blobs.open(resources.blob(grant.blobId()))) {
            if (input.size() != size) {
                throw new IOException("媒体内容大小与登记不符");
            }
            input.position(range.start());
            ByteBuffer buffer = ByteBuffer.allocate((int) Math.min(65536, range.length()));
            // 在发送响应头前认证首个分段，损坏文件不返回貌似成功的媒体响应。
            int first = read(input, buffer);
            if (first < 0) {
                throw new EOFException("媒体内容不完整");
            }
            response.setStatus(range.partial() ? 206 : 200);
            response.setContentType(media.content().mimeType());
            response.setContentLengthLong(range.length());
            response.setHeader(HttpHeaders.ACCEPT_RANGES, "bytes");
            response.setHeader(HttpHeaders.ETAG, etag);
            response.setHeader(HttpHeaders.CACHE_CONTROL, "private, no-store");
            response.setHeader("X-Content-Type-Options", "nosniff");
            response.setHeader("Referrer-Policy", "no-referrer");
            response.setHeader("Content-Security-Policy", "sandbox");
            boolean attachment = grant.download() || media.content().kind() == MediaKind.TEXT;
            response.setHeader(HttpHeaders.CONTENT_DISPOSITION,
                    (attachment ? ContentDisposition.attachment() : ContentDisposition.inline())
                            .filename(media.name(), StandardCharsets.UTF_8).build().toString());
            if (range.partial()) {
                response.setHeader(HttpHeaders.CONTENT_RANGE,
                        "bytes " + range.start() + "-" + range.end() + "/" + size);
            }
            if ("HEAD".equals(request.getMethod())) {
                return;
            }
            var output = response.getOutputStream();
            output.write(buffer.array(), 0, first);
            long remaining = range.length() - first;
            long recheckBytes = first;
            long lastCheck = System.nanoTime();
            while (remaining > 0) {
                if (recheckBytes >= 8 * 1024 * 1024 || System.nanoTime() - lastCheck > 10_000_000_000L) {
                    tickets.authorize(id, ticket);
                    recheckBytes = 0;
                    lastCheck = System.nanoTime();
                }
                buffer.clear().limit((int) Math.min(buffer.capacity(), remaining));
                int count = read(input, buffer);
                if (count < 0) {
                    throw new EOFException("媒体内容不完整");
                }
                output.write(buffer.array(), 0, count);
                remaining -= count;
                recheckBytes += count;
            }
        }
    }

    private int read(SeekableByteChannel input, ByteBuffer target) throws IOException {
        int emptyReads = 0;
        int count;
        while ((count = input.read(target)) == 0) {
            if (++emptyReads > 8) {
                throw new IOException("媒体读取未取得进展");
            }
        }
        return count;
    }
}
