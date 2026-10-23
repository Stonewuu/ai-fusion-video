package com.stonewu.fusion.mapper.team;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.team.Team;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface TeamMapper extends BaseMapper<Team> {
    Team lock(@Param("id") long id);

    Long lockEnabled(@Param("id") long id);
}
