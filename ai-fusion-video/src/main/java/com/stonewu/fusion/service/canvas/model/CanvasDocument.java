package com.stonewu.fusion.service.canvas.model;

import java.time.LocalDateTime;

public record CanvasDocument(String id, String title, Long projectId, int ownerType, long ownerId,
                             long createdBy, int schemaVersion, long revision, long nextEventSequence,
                             boolean archived, LocalDateTime createTime, LocalDateTime updateTime) {
}
