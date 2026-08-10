package com.stonewu.fusion.service.system;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.mapper.system.RoleMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class RoleService {

    private final RoleMapper roleMapper;

    @Transactional
    public Role createRole(String name, String code, Integer sort, String remark) {
        boolean exists = roleMapper.exists(new LambdaQueryWrapper<Role>().eq(Role::getCode, code));
        if (exists) {
            throw new BusinessException(400, "角色编码已存在");
        }
        Role role = Role.builder()
                .name(name).code(code).sort(sort != null ? sort : 0).status(1).remark(remark)
                .build();
        roleMapper.insert(role);
        return role;
    }

    @Transactional
    public void updateRole(Long id, String name, Integer sort, Integer status, String remark) {
        Role role = roleMapper.selectById(id);
        if (role == null) throw new BusinessException(404, "角色不存在");
        if (name != null) role.setName(name);
        if (sort != null) role.setSort(sort);
        if (status != null) role.setStatus(status);
        if (remark != null) role.setRemark(remark);
        roleMapper.updateById(role);
    }

    @Transactional
    public void deleteRole(Long id) {
        roleMapper.deleteById(id);
    }

    public Role getById(Long id) {
        return roleMapper.selectById(id);
    }

    public List<Role> getEnabledRoles() {
        return roleMapper.selectList(new LambdaQueryWrapper<Role>().eq(Role::getStatus, 1));
    }

    public List<Role> getAllRoles() {
        return roleMapper.selectList(null);
    }
}
