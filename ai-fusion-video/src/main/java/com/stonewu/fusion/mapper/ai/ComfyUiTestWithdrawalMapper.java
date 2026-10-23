package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import org.apache.ibatis.annotations.Param;

import com.stonewu.fusion.entity.ai.ComfyUiTestWithdrawalEntity;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ComfyUiTestWithdrawalMapper extends BaseMapper<ComfyUiTestWithdrawalEntity> {

    int recordWithdrawal(@Param("actor") long actor, @Param("request") String request, @Param("team") Long team);
}
