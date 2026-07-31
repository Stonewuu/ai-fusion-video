package com.stonewu.fusion.convert.storyboard;

import com.stonewu.fusion.controller.storyboard.vo.StoryboardEpisodeCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardEpisodeUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardItemCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardItemUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardSceneCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardSceneUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardUpdateReqVO;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.factory.Mappers;

import java.util.List;

/**
 * 分镜 Convert
 */
@Mapper
public interface StoryboardConvert {

    StoryboardConvert INSTANCE = Mappers.getMapper(StoryboardConvert.class);

    // ========== 分镜脚本 ==========

    @Mapping(target = "projectId", ignore = true)
    @Mapping(target = "scriptId", ignore = true)
    @Mapping(target = "title", ignore = true)
    @Mapping(target = "scope", ignore = true)
    @Mapping(target = "ownerType", ignore = true)
    @Mapping(target = "ownerId", ignore = true)
    @Mapping(target = "status", ignore = true)
    Storyboard convert(StoryboardUpdateReqVO reqVO);

    // ========== 分镜集 ==========

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "deletedId", ignore = true)
    @Mapping(target = "composedVideoUrl", ignore = true)
    @Mapping(target = "composeStatus", ignore = true)
    @Mapping(target = "composeErrorMsg", ignore = true)
    @Mapping(target = "composedAt", ignore = true)
    StoryboardEpisode convert(StoryboardEpisodeCreateReqVO reqVO);

    @Mapping(target = "storyboardId", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "deletedId", ignore = true)
    @Mapping(target = "composedVideoUrl", ignore = true)
    @Mapping(target = "composeStatus", ignore = true)
    @Mapping(target = "composeErrorMsg", ignore = true)
    @Mapping(target = "composedAt", ignore = true)
    StoryboardEpisode convert(StoryboardEpisodeUpdateReqVO reqVO);

    // ========== 分镜场次 ==========

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "status", ignore = true)
    StoryboardScene convert(StoryboardSceneCreateReqVO reqVO);

    @Mapping(target = "episodeId", ignore = true)
    @Mapping(target = "storyboardId", ignore = true)
    @Mapping(target = "status", ignore = true)
    StoryboardScene convert(StoryboardSceneUpdateReqVO reqVO);

    // ========== 分镜条目 ==========

    @Mapping(target = "aiGenerated", ignore = true)
    @Mapping(target = "id", ignore = true)
    @Mapping(target = "status", ignore = true)
    StoryboardItem convert(StoryboardItemCreateReqVO reqVO);

    @Mapping(target = "aiGenerated", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "characterIds", ignore = true)
    @Mapping(target = "sceneAssetItemId", ignore = true)
    @Mapping(target = "propIds", ignore = true)
    StoryboardItem convert(StoryboardItemUpdateReqVO reqVO);

    List<StoryboardItem> convertCreateList(List<StoryboardItemCreateReqVO> list);
}
