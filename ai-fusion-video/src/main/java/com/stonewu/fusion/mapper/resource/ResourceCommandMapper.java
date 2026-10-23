package com.stonewu.fusion.mapper.resource;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.resource.ResourceCommandEntity;
import com.stonewu.fusion.entity.resource.ResourceEventEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface ResourceCommandMapper extends BaseMapper<ResourceCommandEntity> {
    int admit(@Param("row") ResourceCommandEntity row);

    ResourceCommandEntity lockRequest(@Param("userId") long userId, @Param("requestId") String requestId);

    int commit(@Param("receipt") String receipt, @Param("undo") String undo, @Param("id") String id);

    int appendEvent(@Param("row") ResourceEventEntity row);

    long head(@Param("lock") boolean lock);

    ResourceEventEntity nextEvent();

    int appendFeed(@Param("sequence") long sequence, @Param("eventId") long eventId);

    int advanceHead(@Param("sequence") long sequence);

    int markPublished(@Param("eventId") long eventId);

    List<String> targetCanvases(@Param("projectId") Long projectId, @Param("type") String type,
                                @Param("resourceId") long resourceId);

    int projectEvent(@Param("canvasId") String canvasId, @Param("sequence") long sequence,
                     @Param("payload") String payload);

    int recordProjection(@Param("eventId") String eventId, @Param("canvasId") String canvasId,
                         @Param("sequence") long sequence);

    List<ResourceEventEntity> feed(@Param("after") long after, @Param("head") long head, @Param("limit") int limit);

    List<String> history(@Param("canvasId") String canvasId, @Param("projectId") Long projectId,
                         @Param("beforeTime") LocalDateTime beforeTime, @Param("beforeId") String beforeId,
                         @Param("limit") int limit);
}
