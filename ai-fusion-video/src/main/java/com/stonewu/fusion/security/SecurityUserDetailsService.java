package com.stonewu.fusion.security;

import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.service.system.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Spring Security UserDetailsService 实现
 */
@Service
@RequiredArgsConstructor
public class SecurityUserDetailsService implements UserDetailsService {

    private final UserService userService;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return loadUserByUsername(username, null);
    }

    public UserDetails loadUserByUsername(String username, Long currentTeamId) throws UsernameNotFoundException {
        User user = userService.getByUsername(username);
        if (user == null) {
            throw new UsernameNotFoundException("用户不存在: " + username);
        }
        List<Role> roles = userService.getUserRoles(user.getId());
        return new SecurityUserDetails(user, roles, currentTeamId);
    }
}
