package com.stonewu.fusion.mapper.storyboard;

import java.util.List;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface StoryboardItemMapper extends BaseMapper<StoryboardItem> {
    long totalDurationSeconds(@Param("storyboardId") long storyboardId);

    List<Long> composeClipIds(@Param("episodeId") long episodeId);
}
