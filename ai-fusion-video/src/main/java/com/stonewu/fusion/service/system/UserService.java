package com.stonewu.fusion.service.system;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.entity.system.UserRole;
import com.stonewu.fusion.enums.team.TeamMemberRoleEnum;
import com.stonewu.fusion.mapper.system.RoleMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.system.UserRoleMapper;
import com.stonewu.fusion.service.team.TeamService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class UserService {

    private static final String ADMIN_ROLE_CODE = "admin";
    private static final String USER_ROLE_CODE = "user";

    private final UserMapper userMapper;
    private final RoleMapper roleMapper;
    private final UserRoleMapper userRoleMapper;
    private final PasswordEncoder passwordEncoder;
    private final SystemConfigService systemConfigService;
    private final TeamService teamService;

    @Transactional
    public User register(String username, String password, String nickname) {
        if (!isInitialized()) {
            throw new BusinessException(400, "系统尚未初始化，请先完成管理员初始化");
        }
        if (!systemConfigService.isRegistrationEnabled()) {
            throw new BusinessException(400, "系统未开放注册");
        }
        User user = createUser(username, password, nickname);
        assignRoleIfPresent(user.getId(), USER_ROLE_CODE);
        teamService.addUserToSingleTeam(user.getId(), TeamMemberRoleEnum.MEMBER.getRole());
        return user;
    }

    @Transactional
    public User initializeAdmin(String username, String password, String nickname) {
        if (isInitialized()) {
            throw new BusinessException(400, "系统已初始化，请勿重复操作");
        }
        User user = createUser(username, password, nickname);
        assignRoleIfPresent(user.getId(), ADMIN_ROLE_CODE);
        assignRoleIfPresent(user.getId(), USER_ROLE_CODE);
        teamService.createInitialTeam(username + "的团队", user.getId());
        systemConfigService.setRegistrationEnabled(false);
        return user;
    }

    public boolean isInitialized() {
        Role adminRole = roleMapper.selectOne(new LambdaQueryWrapper<Role>().eq(Role::getCode, ADMIN_ROLE_CODE));
        if (adminRole == null) {
            throw new BusinessException(500, "系统角色数据异常，请检查数据库初始化");
        }
        return userRoleMapper.exists(new LambdaQueryWrapper<UserRole>().eq(UserRole::getRoleId, adminRole.getId()));
    }

    @Cacheable(value = "userByUsername", key = "#username")
    public User getByUsername(String username) {
        return userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getUsername, username));
    }

    public User getById(Long id) {
        return userMapper.selectById(id);
    }

    public PageResult<User> getPage(String username, String nickname, Integer status, int pageNo, int pageSize) {
        LambdaQueryWrapper<User> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(username != null, User::getUsername, username)
                .like(nickname != null, User::getNickname, nickname)
                .eq(status != null, User::getStatus, status)
                .orderByDesc(User::getId);
        return PageResult.of(userMapper.selectPage(new Page<>(pageNo, pageSize), wrapper));
    }

    @Transactional
    @CacheEvict(value = "userByUsername", allEntries = true)
    public void updateUser(Long id, String nickname, String avatar, String email, String phone, Integer status) {
        User user = userMapper.selectById(id);
        if (user == null)
            throw new BusinessException(404, "用户不存在");
        if (nickname != null)
            user.setNickname(nickname);
        if (avatar != null)
            user.setAvatar(avatar);
        if (email != null)
            user.setEmail(email);
        if (phone != null)
            user.setPhone(phone);
        if (status != null)
            user.setStatus(status);
        userMapper.updateById(user);
    }

    /**
     * 当前用户修改自己的个人资料
     */
    @Transactional
    @CacheEvict(value = "userByUsername", allEntries = true)
    public void updateProfile(Long userId, String nickname, String email, String phone) {
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(404, "用户不存在");
        }
        if (nickname != null) {
            user.setNickname(nickname);
        }
        if (email != null) {
            user.setEmail(email);
        }
        if (phone != null) {
            user.setPhone(phone);
        }
        userMapper.updateById(user);
    }

    /**
     * 修改密码（需验证旧密码）
     */
    @Transactional
    @CacheEvict(value = "userByUsername", allEntries = true)
    public void changePassword(Long userId, String oldPassword, String newPassword) {
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(404, "用户不存在");
        }
        if (!passwordEncoder.matches(oldPassword, user.getPassword())) {
            throw new BusinessException(400, "旧密码不正确");
        }
        user.setPassword(passwordEncoder.encode(newPassword));
        userMapper.updateById(user);
    }

    @Transactional
    @CacheEvict(value = "userByUsername", allEntries = true)
    public void deleteUser(Long id) {
        userMapper.deleteById(id);
    }

    @Cacheable(value = "userRoles", key = "#userId")
    public List<Role> getUserRoles(Long userId) {
        List<UserRole> userRoles = userRoleMapper.selectList(
                new LambdaQueryWrapper<UserRole>().eq(UserRole::getUserId, userId));
        List<Long> roleIds = userRoles.stream().map(UserRole::getRoleId).toList();
        if (roleIds.isEmpty()) {
            return List.of();
        }
        return roleMapper.selectByIds(roleIds);
    }

    @Transactional
    @CacheEvict(value = "userRoles", key = "#userId")
    public void assignRole(Long userId, Long roleId) {
        boolean exists = userRoleMapper.exists(
                new LambdaQueryWrapper<UserRole>()
                        .eq(UserRole::getUserId, userId)
                        .eq(UserRole::getRoleId, roleId));
        if (!exists) {
            userRoleMapper.insert(UserRole.builder().userId(userId).roleId(roleId).build());
        }
    }

    @Transactional
    @CacheEvict(value = "userRoles", key = "#userId")
    public void removeRole(Long userId, Long roleId) {
        userRoleMapper.delete(new LambdaQueryWrapper<UserRole>()
                .eq(UserRole::getUserId, userId)
                .eq(UserRole::getRoleId, roleId));
    }

    private User createUser(String username, String password, String nickname) {
        boolean exists = userMapper.exists(new LambdaQueryWrapper<User>().eq(User::getUsername, username));
        if (exists) {
            throw new BusinessException(400, "用户名已存在");
        }
        User user = User.builder()
                .username(username)
                .password(passwordEncoder.encode(password))
                .nickname(nickname != null ? nickname : username)
                .status(1)
                .build();
        userMapper.insert(user);
        return user;
    }

    private void assignRoleIfPresent(Long userId, String roleCode) {
        Role role = roleMapper.selectOne(new LambdaQueryWrapper<Role>().eq(Role::getCode, roleCode));
        if (role != null) {
            userRoleMapper.insert(UserRole.builder().userId(userId).roleId(role.getId()).build());
        }
    }

    public User getByUsernameOrEmail(String usernameOrEmail) {
        return userMapper.selectOne(new LambdaQueryWrapper<User>()
                .eq(User::getUsername, usernameOrEmail)
                .or()
                .eq(User::getEmail, usernameOrEmail));
    }

    @Transactional
    @CacheEvict(value = "userByUsername", allEntries = true)
    public void resetPassword(Long userId, String newPassword) {
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(404, "用户不存在");
        }
        user.setPassword(passwordEncoder.encode(newPassword));
        userMapper.updateById(user);
    }

    public boolean isAdmin(Long userId) {
        Role adminRole = roleMapper.selectOne(new LambdaQueryWrapper<Role>().eq(Role::getCode, ADMIN_ROLE_CODE));
        if (adminRole == null) {
            return false;
        }
        return userRoleMapper.exists(new LambdaQueryWrapper<UserRole>()
                .eq(UserRole::getUserId, userId)
                .eq(UserRole::getRoleId, adminRole.getId()));
    }
}
