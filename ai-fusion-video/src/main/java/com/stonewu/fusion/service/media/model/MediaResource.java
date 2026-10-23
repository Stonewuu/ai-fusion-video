package com.stonewu.fusion.service.media.model;

import java.time.LocalDateTime;

/**
 * 可返回客户端的资源投影，不包含访问凭据、密钥或存储配置。
 */
public record MediaResource(String id, Long projectId, int ownerType, long ownerId, long createdBy,
                            String name, MediaFacts content, String state, int version,
                            String lastError, LocalDateTime createdAt, LocalDateTime updatedAt) {
}
