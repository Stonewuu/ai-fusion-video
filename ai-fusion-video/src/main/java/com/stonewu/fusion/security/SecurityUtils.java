package com.stonewu.fusion.security;

import com.stonewu.fusion.common.BusinessException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;

/**
 * 安全工具类：从 SecurityContext 获取当前登录用户信息
 */
public final class SecurityUtils {

    private SecurityUtils() {
    }

    /**
     * 获取当前登录用户 ID
     */
    public static Long getCurrentUserId() {
        SecurityUserDetails userDetails = getCurrentUserDetails();
        return userDetails != null ? userDetails.getUserId() : null;
    }

    /**
     * 获取当前登录用户名
     */
    public static String getCurrentUsername() {
        SecurityUserDetails userDetails = getCurrentUserDetails();
        return userDetails != null ? userDetails.getUsername() : null;
    }

    /**
     * 获取当前认证中的团队 ID
     */
    public static Long getCurrentTeamId() {
        SecurityUserDetails userDetails = getCurrentUserDetails();
        return userDetails != null ? userDetails.getCurrentTeamId() : null;
    }

    /**
     * 获取当前登录用户的 SecurityUserDetails
     */
    public static SecurityUserDetails getCurrentUserDetails() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()) {
            return null;
        }
        Object principal = authentication.getPrincipal();
        if (principal instanceof SecurityUserDetails details) {
            return details;
        }
        return null;
    }

    /**
     * 获取当前登录用户 ID（必须已登录，否则抛异常）
     */
    public static Long requireCurrentUserId() {
        Long userId = getCurrentUserId();
        if (userId == null) {
            throw new BusinessException("用户未登录");
        }
        return userId;
    }

    /**
     * 获取当前认证中的团队 ID（必须存在，否则抛异常）
     */
    public static Long requireCurrentTeamId() {
        Long teamId = getCurrentTeamId();
        if (teamId == null) {
            throw new BusinessException("当前会话未绑定团队");
        }
        return teamId;
    }

    /**
     * 刷新当前认证中的团队 ID
     */
    public static void refreshCurrentTeamId(Long teamId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        SecurityUserDetails userDetails = getCurrentUserDetails();
        if (authentication == null || userDetails == null) {
            return;
        }
        SecurityUserDetails updatedDetails = userDetails.withCurrentTeamId(teamId);
        UsernamePasswordAuthenticationToken updatedAuthentication = new UsernamePasswordAuthenticationToken(
                updatedDetails,
                authentication.getCredentials(),
                updatedDetails.getAuthorities());
        updatedAuthentication.setDetails(authentication.getDetails());
        SecurityContextHolder.getContext().setAuthentication(updatedAuthentication);
    }
}
