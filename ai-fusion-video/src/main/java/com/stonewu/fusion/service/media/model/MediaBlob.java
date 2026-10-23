package com.stonewu.fusion.service.media.model;

import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * 不可序列化密钥，不使用 record 或 Lombok 生成包含密钥的 toString。
 */
public final class MediaBlob {
    private final String id;
    private final String resourceId;
    private final long storageConfigId;
    private final String backend;
    private final String objectKey;
    private final byte[] keyset;
    private final Long storedBytes;

    public MediaBlob(String id, String resourceId, long storageConfigId, String backend,
                     String objectKey, byte[] keyset, Long storedBytes) {
        this.id = id;
        this.resourceId = resourceId;
        this.storageConfigId = storageConfigId;
        this.backend = backend;
        this.objectKey = objectKey;
        this.keyset = keyset.clone();
        this.storedBytes = storedBytes;
    }

    public String id() {
        return id;
    }

    public String resourceId() {
        return resourceId;
    }

    public long storageConfigId() {
        return storageConfigId;
    }

    public String backend() {
        return backend;
    }

    public String objectKey() {
        return objectKey;
    }

    @JsonIgnore
    public byte[] keyset() {
        return keyset.clone();
    }

    public Long storedBytes() {
        return storedBytes;
    }

    @Override
    public String toString() {
        return "MediaBlob[id=" + id + ", resourceId=" + resourceId + "]";
    }
}
