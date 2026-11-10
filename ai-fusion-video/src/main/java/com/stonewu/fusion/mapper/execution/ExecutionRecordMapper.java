package com.stonewu.fusion.mapper.execution;

import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Kind;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface ExecutionRecordMapper {
    record Row(Kind kind, String id, String title, String status, String message,
               LocalDateTime createdAt, LocalDateTime updatedAt) {
    }

    List<Row> list(@Param("userId") long userId, @Param("statuses") List<String> statuses,
                   @Param("since") LocalDateTime since, @Param("offset") int offset, @Param("limit") int limit);

    long count(@Param("userId") long userId, @Param("statuses") List<String> statuses,
               @Param("since") LocalDateTime since);

    Row get(@Param("userId") long userId, @Param("kind") Kind kind, @Param("id") String id);
}
