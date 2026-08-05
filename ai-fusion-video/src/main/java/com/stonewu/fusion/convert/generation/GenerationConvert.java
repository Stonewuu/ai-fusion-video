package com.stonewu.fusion.convert.generation;

import com.stonewu.fusion.controller.generation.vo.ImageTaskSubmitReqVO;
import com.stonewu.fusion.controller.generation.vo.VideoTaskSubmitReqVO;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.factory.Mappers;

/**
 * 生成任务 Convert
 */
@Mapper
public interface GenerationConvert {

    GenerationConvert INSTANCE = Mappers.getMapper(GenerationConvert.class);

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "taskId", ignore = true)
    @Mapping(target = "userId", ignore = true)
    @Mapping(target = "successCount", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "errorMsg", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    @Mapping(target = "workflowVersionId", ignore = true)
    ImageTask convert(ImageTaskSubmitReqVO reqVO);

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "taskId", ignore = true)
    @Mapping(target = "userId", ignore = true)
    @Mapping(target = "successCount", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "errorMsg", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    @Mapping(target = "workflowVersionId", ignore = true)
    VideoTask convert(VideoTaskSubmitReqVO reqVO);
}
