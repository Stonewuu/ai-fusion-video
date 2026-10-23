package com.stonewu.fusion.service.media.model;

/**
 * 对象接收凭证只在服务器传递；每次重试写入独立对象，旧工作者不能覆盖新结果。
 */
public record MediaIngestClaim(MediaResource resource, String blobId, long epoch, boolean replay,
                               long actorUserId, MediaSourceSnapshot source, MediaGenerationOrigin generation,
                               MediaJobOrigin job) {
    public MediaIngestClaim(MediaResource resource, String blobId, long epoch, boolean replay,
                            long actorUserId, MediaSourceSnapshot source, MediaGenerationOrigin generation) {
        this(resource, blobId, epoch, replay, actorUserId, source, generation, null);
    }

    public MediaIngestClaim(MediaResource resource, String blobId, long epoch, boolean replay,
                            long actorUserId, MediaSourceSnapshot source) {
        this(resource, blobId, epoch, replay, actorUserId, source, null);
    }

    public MediaIngestClaim(MediaResource resource, String blobId, long epoch, boolean replay) {
        this(resource, blobId, epoch, replay, resource.createdBy(), null);
    }
}
