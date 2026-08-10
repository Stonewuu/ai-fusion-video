package com.stonewu.fusion.convert.system;

import com.stonewu.fusion.controller.system.vo.RoleRespVO;
import com.stonewu.fusion.entity.system.Role;
import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;

import java.util.List;

/**
 * 角色 Convert
 */
@Mapper
public interface RoleConvert {

    RoleConvert INSTANCE = Mappers.getMapper(RoleConvert.class);

    RoleRespVO convert(Role role);

    List<RoleRespVO> convertList(List<Role> roles);
}
