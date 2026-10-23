package com.stonewu.fusion.mapper.storyboard;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface StoryboardMapper extends BaseMapper<Storyboard> {
    Long lockLatest(@Param("projectId") long projectId, @Param("scriptId") long scriptId);
}
