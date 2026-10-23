package com.stonewu.fusion.mapper.system;

import java.util.List;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.system.User;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface UserMapper extends BaseMapper<User> {
    Long lockEnabled(@Param("id") long id);

    Long lock(@Param("id") long id);

    String displayName(@Param("id") long id);

    List<Long> lockPair(@Param("first") long first, @Param("second") long second);
}
