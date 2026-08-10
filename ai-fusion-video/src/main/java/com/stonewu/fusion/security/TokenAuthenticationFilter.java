package com.stonewu.fusion.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * Token 认证过滤器
 * <p>
 * 从请求头中提取 access_token，到 Redis 中验证后设置 SecurityContext
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class TokenAuthenticationFilter extends OncePerRequestFilter {

    private final TokenService tokenService;
    private final SecurityUserDetailsService userDetailsService;

    @Override
    protected void doFilterInternal(HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {
        try {
            String token = getTokenFromRequest(request);
            if (StringUtils.hasText(token)) {
                TokenService.TokenSession session = tokenService.getAccessTokenSession(token);
                if (session != null) {
                    String username = session.getUsername();
                    if (username != null) {
                        UserDetails userDetails = userDetailsService.loadUserByUsername(username, session.getCurrentTeamId());

                        UsernamePasswordAuthenticationToken authentication = new UsernamePasswordAuthenticationToken(
                                userDetails, null, userDetails.getAuthorities());
                        authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                        SecurityContextHolder.getContext().setAuthentication(authentication);

                        // access_token 固定有效期，不再自动续期，由前端通过 refresh_token 刷新
                    }
                }
            }
        } catch (Throwable e) {
            log.error("Token authentication failed: {}", e.getMessage(), e);
        }
        filterChain.doFilter(request, response);
    }

    private String getTokenFromRequest(HttpServletRequest request) {
        // 支持 Authorization: Bearer {token} 和 Authorization: {token} 两种方式
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken)) {
            if (bearerToken.startsWith("Bearer ")) {
                return bearerToken.substring(7);
            }
            return bearerToken;
        }
        // 也支持从请求参数中获取
        return request.getParameter("access_token");
    }
}
