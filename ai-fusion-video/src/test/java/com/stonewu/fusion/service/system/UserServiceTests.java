package com.stonewu.fusion.service.system;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.entity.system.UserRole;
import com.stonewu.fusion.enums.team.TeamMemberRoleEnum;
import com.stonewu.fusion.mapper.system.RoleMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.system.UserRoleMapper;
import com.stonewu.fusion.service.team.TeamService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserServiceTests {

    @Mock
    private UserMapper userMapper;

    @Mock
    private RoleMapper roleMapper;

    @Mock
    private UserRoleMapper userRoleMapper;

    @Mock
    private PasswordEncoder passwordEncoder;

    @Mock
    private SystemConfigService systemConfigService;

    @Mock
    private TeamService teamService;

    @InjectMocks
    private UserService userService;

    @Test
    void registerShouldRejectWhenSystemNotInitialized() {
        mockAdminRole();
        when(userRoleMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(false);

        BusinessException exception = assertThrows(BusinessException.class,
                () -> userService.register("stone", "secret123", null));

        assertEquals(400, exception.getCode());
        assertEquals("系统尚未初始化，请先完成管理员初始化", exception.getMessage());
        verify(userMapper, never()).insert(any(User.class));
    }

    @Test
    void registerShouldRejectWhenRegistrationDisabled() {
        mockAdminRole();
        when(userRoleMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(true);
        when(systemConfigService.isRegistrationEnabled()).thenReturn(false);

        BusinessException exception = assertThrows(BusinessException.class,
                () -> userService.register("stone", "secret123", null));

        assertEquals(400, exception.getCode());
        assertEquals("系统未开放注册", exception.getMessage());
        verify(userMapper, never()).insert(any(User.class));
    }

    @Test
    void registerShouldCreateUserAndJoinSingleTeam() {
        mockUserCreation();
        mockAdminAndUserRoles();
        when(userRoleMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(true);
        when(systemConfigService.isRegistrationEnabled()).thenReturn(true);
        when(userMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(false);

        User user = userService.register("stone", "secret123", null);

        assertEquals(100L, user.getId());
        assertEquals("stone", user.getUsername());
        assertEquals("stone", user.getNickname());

        ArgumentCaptor<UserRole> roleCaptor = ArgumentCaptor.forClass(UserRole.class);
        verify(userRoleMapper).insert(roleCaptor.capture());
        assertEquals(2L, roleCaptor.getValue().getRoleId());
        verify(teamService).addUserToSingleTeam(100L, TeamMemberRoleEnum.MEMBER.getRole());
    }

    @Test
    void initializeAdminShouldCreateAdminAndInitialTeam() {
        mockUserCreation();
        mockInitAdminRoles();
        when(userRoleMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(false);
        when(userMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(false);

        User user = userService.initializeAdmin("admin", "secret123", "管理员");

        assertEquals(100L, user.getId());
        assertEquals("管理员", user.getNickname());
        ArgumentCaptor<UserRole> roleCaptor = ArgumentCaptor.forClass(UserRole.class);
        verify(userRoleMapper, times(2)).insert(roleCaptor.capture());
        List<Long> roleIds = roleCaptor.getAllValues().stream().map(UserRole::getRoleId).toList();
        assertEquals(List.of(1L, 2L), roleIds);
        verify(teamService).createInitialTeam("admin的团队", 100L);
        verify(systemConfigService).setRegistrationEnabled(false);
    }

    @Test
    void initializeAdminShouldRejectDuplicateUsername() {
        mockAdminRole();
        when(userRoleMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(false);
        when(userMapper.exists(any(LambdaQueryWrapper.class))).thenReturn(true);

        BusinessException exception = assertThrows(BusinessException.class,
                () -> userService.initializeAdmin("admin", "secret123", null));

        assertEquals(400, exception.getCode());
        assertTrue(exception.getMessage().contains("用户名已存在"));
    }

    private void mockAdminRole() {
        Role adminRole = Role.builder().id(1L).code("admin").name("管理员").build();
        when(roleMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(adminRole);
    }

    private void mockAdminAndUserRoles() {
        Role adminRole = Role.builder().id(1L).code("admin").name("管理员").build();
        Role userRole = Role.builder().id(2L).code("user").name("普通用户").build();
        when(roleMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(adminRole, userRole);
    }

    private void mockInitAdminRoles() {
        Role adminRole = Role.builder().id(1L).code("admin").name("管理员").build();
        Role userRole = Role.builder().id(2L).code("user").name("普通用户").build();
        when(roleMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(adminRole, adminRole, userRole);
    }

    private void mockUserCreation() {
        doAnswer(invocation -> {
            User user = invocation.getArgument(0);
            user.setId(100L);
            return 1;
        }).when(userMapper).insert(any(User.class));
        when(passwordEncoder.encode("secret123")).thenReturn("encoded-secret123");
    }
}