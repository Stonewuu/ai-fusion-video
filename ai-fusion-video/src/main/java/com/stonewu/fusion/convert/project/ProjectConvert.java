package com.stonewu.fusion.convert.project;

import com.stonewu.fusion.controller.project.vo.ProjectCreateReqVO;
import com.stonewu.fusion.controller.project.vo.ProjectUpdateReqVO;
import com.stonewu.fusion.entity.project.Project;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.factory.Mappers;

/**
 * 项目 Convert
 */
@Mapper
public interface ProjectConvert {

    ProjectConvert INSTANCE = Mappers.getMapper(ProjectConvert.class);

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "scope", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "artStyle", ignore = true)
    @Mapping(target = "artStyleDescription", ignore = true)
    @Mapping(target = "artStyleImagePrompt", ignore = true)
    @Mapping(target = "artStyleImageUrl", ignore = true)
    Project convert(ProjectCreateReqVO reqVO);

    @Mapping(target = "scope", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    Project convert(ProjectUpdateReqVO reqVO);
}
