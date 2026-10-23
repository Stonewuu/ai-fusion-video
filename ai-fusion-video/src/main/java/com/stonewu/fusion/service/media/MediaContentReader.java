package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

@Component
@RequiredArgsConstructor
public class MediaContentReader {
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final MediaBlobStore blobs;

    /**
     * 服务端处理同样验证实际用户权限，不能通过内部存储路径绕过媒体授权。
     */
    public void copy(String id, CanvasActor actor, Path destination, long limit) throws IOException {
        copy(id, actor, destination, limit, () -> {
        });
    }

    public void copy(String id, CanvasActor actor, Path destination, long limit, Runnable progress) throws IOException {
        progress.run();
        var row = resources.get(id, false);
        access.ready(row.resource(), actor);
        if (row.resource().content().byteSize() > limit) {
            throw new MediaException(413, "MEDIA_TOO_LARGE", "媒体超过本操作允许的大小");
        }
        try (var input = blobs.open(resources.blob(row.blobId())); var output = Files.newOutputStream(destination)) {
            ByteBuffer buffer = ByteBuffer.allocate(65536);
            long count = 0;
            long lastCheck = System.nanoTime();
            int emptyReads = 0;
            while (true) {
                int read = input.read(buffer);
                if (read < 0) {
                    break;
                }
                if (read == 0) {
                    if (++emptyReads > 8) {
                        throw new IOException("媒体读取没有进展");
                    }
                    continue;
                }
                emptyReads = 0;
                count += read;
                if (count > limit) {
                    throw new MediaException(413, "MEDIA_TOO_LARGE", "媒体超过本操作允许的大小");
                }
                output.write(buffer.array(), 0, read);
                buffer.clear();
                if (System.nanoTime() - lastCheck >= 5_000_000_000L) {
                    progress.run();
                    access.ready(resources.get(id, false).resource(), actor);
                    lastCheck = System.nanoTime();
                }
            }
            if (count != row.resource().content().byteSize()) {
                throw new IOException("媒体内容长度不符");
            }
        }
        access.ready(resources.get(id, false).resource(), actor);
        progress.run();
    }
}
