package com.stonewu.fusion.mapper.canvas;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import java.util.List;
import java.util.Map;

import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationData.Person;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import org.apache.ibatis.annotations.Param;

import com.stonewu.fusion.entity.canvas.CanvasMemberEntity;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMemberMapper extends BaseMapper<CanvasMemberEntity> {

    IPage<Person> people(Page<Person> page, @Param("canvasId") String canvasId, @Param("projectId") Long projectId);

    Person findPerson(@Param("username") String username);

    Person person(@Param("userId") long userId);

    int grant(@Param("canvasId") String canvasId, @Param("userId") long userId, @Param("canEdit") boolean canEdit,
              @Param("canExecute") boolean canExecute, @Param("canManage") boolean canManage,
              @Param("actor") long actor);

    List<Map<String, Object>> projectGrants(@Param("projectId") long projectId, @Param("userId") long userId);

    int removeProjectGrants(@Param("projectId") long projectId, @Param("userId") long userId);
}
