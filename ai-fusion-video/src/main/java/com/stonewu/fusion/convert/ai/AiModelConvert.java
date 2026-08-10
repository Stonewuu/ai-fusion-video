package com.stonewu.fusion.convert.ai;

import com.stonewu.fusion.controller.ai.vo.AiModelRespVO;
import com.stonewu.fusion.entity.ai.AiModel;
import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;

import java.util.List;

/**
 * AI模型 Convert
 */
@Mapper
public interface AiModelConvert {

    AiModelConvert INSTANCE = Mappers.getMapper(AiModelConvert.class);

    AiModelRespVO convert(AiModel model);

    List<AiModelRespVO> convertList(List<AiModel> models);
}
