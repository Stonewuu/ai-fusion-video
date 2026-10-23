package com.stonewu.fusion.service.media.model;

/**
 * URL 仅在服务器中用于接收，不能作为工具/审计结果输出，可能包含旧存储的签名参数。
 */
public record MediaSourceSnapshot(MediaSourceReference reference, Long projectId, int ownerType,
                                  long ownerId, String name, String uri, MediaKind kind) {
    @Override
    public String toString() {
        return "MediaSourceSnapshot[reference=" + reference + "]";
    }
}
