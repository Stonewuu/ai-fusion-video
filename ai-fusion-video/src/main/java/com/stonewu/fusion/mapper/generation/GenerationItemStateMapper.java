package com.stonewu.fusion.mapper.generation;

import com.stonewu.fusion.service.generation.GenerationKind;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface GenerationItemStateMapper {
    int failPending(@Param("kind") GenerationKind kind, @Param("id") long id, @Param("message") String message);

    int failTaskItem(@Param("kind") GenerationKind kind, @Param("id") long id,
                     @Param("taskId") long taskId, @Param("message") String message);

    int resetFailed(@Param("kind") GenerationKind kind, @Param("id") long id);
}
