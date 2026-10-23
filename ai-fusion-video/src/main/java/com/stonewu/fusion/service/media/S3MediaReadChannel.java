package com.stonewu.fusion.service.media;

import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.NonWritableChannelException;
import java.nio.channels.SeekableByteChannel;

/**
 * 有界分段读取，不把整段视频载入堆内存；对象在整个读取过程中必须保持同一 ETag。
 */
final class S3MediaReadChannel implements SeekableByteChannel {
    private static final int BLOCK_SIZE = 1024 * 1024;
    private final S3Client client;
    private final String bucket;
    private final String key;
    private final String etag;
    private final long size;
    private boolean open = true;
    private long position;
    private long cachedStart = -1;
    private byte[] cached = new byte[0];

    S3MediaReadChannel(S3Client client, String bucket, String key, long expectedSize) throws IOException {
        this.client = client;
        this.bucket = bucket;
        this.key = key;
        try {
            var head = client.headObject(request -> request.bucket(bucket).key(key));
            size = head.contentLength();
            etag = head.eTag();
            if (size != expectedSize || etag == null) {
                throw new IOException("托管媒体对象长度或版本不符");
            }
        } catch (RuntimeException failure) {
            throw new IOException("无法读取托管媒体对象", failure);
        }
    }

    @Override
    public int read(ByteBuffer target) throws IOException {
        ensureOpen();
        if (!target.hasRemaining()) {
            return 0;
        }
        if (position >= size) {
            return -1;
        }
        if (cachedStart < 0 || position < cachedStart || position >= cachedStart + cached.length) {
            fetch();
        }
        int count = (int) Math.min(target.remaining(), cachedStart + cached.length - position);
        target.put(cached, (int) (position - cachedStart), count);
        position += count;
        return count;
    }

    private void fetch() throws IOException {
        long start = (position / BLOCK_SIZE) * BLOCK_SIZE;
        long end = Math.min(size - 1, start + BLOCK_SIZE - 1);
        int length = (int) (end - start + 1);
        var request = GetObjectRequest.builder().bucket(bucket).key(key).ifMatch(etag)
                .range("bytes=" + start + "-" + end).build();
        try (var response = client.getObject(request)) {
            String range = "bytes " + start + "-" + end + "/" + size;
            if (!range.equals(response.response().contentRange()) || response.response().contentLength() != length) {
                response.abort();
                throw new IOException("媒体存储未按范围返回内容");
            }
            byte[] bytes = response.readNBytes(length + 1);
            if (bytes.length != length) {
                response.abort();
                throw new IOException("媒体存储返回了不完整内容");
            }
            cached = bytes;
            cachedStart = start;
        } catch (RuntimeException failure) {
            throw new IOException("读取媒体分段失败", failure);
        }
    }

    @Override
    public long position() throws IOException {
        ensureOpen();
        return position;
    }

    @Override
    public SeekableByteChannel position(long value) throws IOException {
        ensureOpen();
        if (value < 0) {
            throw new IllegalArgumentException("读取位置不能为负数");
        }
        position = value;
        return this;
    }

    @Override
    public long size() throws IOException {
        ensureOpen();
        return size;
    }

    @Override
    public int write(ByteBuffer source) {
        throw new NonWritableChannelException();
    }

    @Override
    public SeekableByteChannel truncate(long value) {
        throw new NonWritableChannelException();
    }

    @Override
    public boolean isOpen() {
        return open;
    }

    @Override
    public void close() {
        open = false;
        cached = new byte[0];
    }

    private void ensureOpen() throws ClosedChannelException {
        if (!open) {
            throw new ClosedChannelException();
        }
    }
}
