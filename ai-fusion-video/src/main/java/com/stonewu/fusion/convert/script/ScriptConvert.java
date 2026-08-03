package com.stonewu.fusion.convert.script;

import com.stonewu.fusion.controller.script.vo.EpisodeCreateReqVO;
import com.stonewu.fusion.controller.script.vo.EpisodeUpdateReqVO;
import com.stonewu.fusion.controller.script.vo.SceneCreateReqVO;
import com.stonewu.fusion.controller.script.vo.SceneUpdateReqVO;
import com.stonewu.fusion.controller.script.vo.ScriptUpdateReqVO;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import org.mapstruct.Builder;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;
import org.mapstruct.factory.Mappers;

/**
 * 剧本 Convert
 */
@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, builder = @Builder(disableBuilder = true))
public interface ScriptConvert {

    ScriptConvert INSTANCE = Mappers.getMapper(ScriptConvert.class);

    // ========== 剧本 ==========

    Script convert(ScriptUpdateReqVO reqVO);

    // ========== 分集 ==========

    ScriptEpisode convert(EpisodeCreateReqVO reqVO);

    ScriptEpisode convert(EpisodeUpdateReqVO reqVO);

    // ========== 场次 ==========

    ScriptSceneItem convert(SceneCreateReqVO reqVO);

    ScriptSceneItem convert(SceneUpdateReqVO reqVO);
}
