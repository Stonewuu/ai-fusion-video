package com.stonewu.fusion.security;

import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.entity.system.User;
import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Spring Security UserDetails 实现
 */
@Getter
public class SecurityUserDetails implements UserDetails {

    private final Long userId;
    private final String username;
    private final String password;
    private final Integer status;
    private final Long currentTeamId;
    private final Collection<? extends GrantedAuthority> authorities;

    public SecurityUserDetails(Long userId,
            String username,
            String password,
            Integer status,
            Long currentTeamId,
            Collection<? extends GrantedAuthority> authorities) {
        this.userId = userId;
        this.username = username;
        this.password = password;
        this.status = status;
        this.currentTeamId = currentTeamId;
        this.authorities = authorities;
    }

    public SecurityUserDetails(User user, List<Role> roles) {
        this(user, roles, null);
    }

    public SecurityUserDetails(User user, List<Role> roles, Long currentTeamId) {
        this.userId = user.getId();
        this.username = user.getUsername();
        this.password = user.getPassword();
        this.status = user.getStatus();
        this.currentTeamId = currentTeamId;
        this.authorities = roles.stream()
                .map(role -> new SimpleGrantedAuthority("ROLE_" + role.getCode().toUpperCase()))
                .collect(Collectors.toList());
    }

    public SecurityUserDetails withCurrentTeamId(Long teamId) {
        return new SecurityUserDetails(userId, username, password, status, teamId, authorities);
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return status != null && status == 1;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return status != null && status == 1;
    }
}
