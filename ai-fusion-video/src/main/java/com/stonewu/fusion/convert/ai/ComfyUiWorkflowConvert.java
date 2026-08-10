package com.stonewu.fusion.convert.ai;

import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowRespVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowVersionRespVO;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;

import java.util.List;

@Mapper
public interface ComfyUiWorkflowConvert {

    ComfyUiWorkflowConvert INSTANCE = Mappers.getMapper(ComfyUiWorkflowConvert.class);

    ComfyUiWorkflowRespVO convert(ComfyUiWorkflow workflow);

    ComfyUiWorkflowVersionRespVO convertVersion(ComfyUiWorkflowVersion version);

    List<ComfyUiWorkflowRespVO> convertList(List<ComfyUiWorkflow> workflows);

    List<ComfyUiWorkflowVersionRespVO> convertVersionList(List<ComfyUiWorkflowVersion> versions);
}
