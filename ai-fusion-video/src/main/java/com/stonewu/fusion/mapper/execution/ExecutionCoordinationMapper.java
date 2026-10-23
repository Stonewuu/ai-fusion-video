package com.stonewu.fusion.mapper.execution;

import org.apache.ibatis.annotations.Mapper;

import java.time.LocalDateTime;

@Mapper
public interface ExecutionCoordinationMapper {
    LocalDateTime now();

    Integer lockGenerationDispatch();
}
