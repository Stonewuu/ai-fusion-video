package com.stonewu.fusion.mapper.ai;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ComfyUiWorkflowMapper extends BaseMapper<ComfyUiWorkflow> {

    int softDeleteById(@Param("id") Long id);

    Long lock(@Param("id") long id);

    ComfyUiWorkflow selectForUpdate(@Param("id") long id);
}
