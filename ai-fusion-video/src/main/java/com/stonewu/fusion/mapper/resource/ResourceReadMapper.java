package com.stonewu.fusion.mapper.resource;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.Map;
import java.util.List;

@Mapper
public interface ResourceReadMapper {
    record Header(long id, int version, boolean deleted, Long projectId) {
    }

    Header header(@Param("type") CanvasValueType type, @Param("id") long id,
                  @Param("historical") boolean historical, @Param("includeDeleted") boolean includeDeleted,
                  @Param("lock") boolean lock);

    Map<String, Object> data(@Param("type") CanvasValueType type, @Param("id") long id, @Param("version") int version);

    Map<String, Object> raw(@Param("type") CanvasValueType type, @Param("id") long id, @Param("lock") boolean lock);

    Long currentStoryboard(@Param("projectId") long projectId);

    List<Long> storyboardChildren(@Param("type") CanvasValueType type, @Param("parentType") CanvasValueType parentType,
                                  @Param("parentId") long parentId);

    long countStoryboardChildren(@Param("type") CanvasValueType type, @Param("storyboardId") long storyboardId);

    List<Header> lockProjectResources(@Param("type") CanvasValueType type, @Param("projectId") long projectId);
}
