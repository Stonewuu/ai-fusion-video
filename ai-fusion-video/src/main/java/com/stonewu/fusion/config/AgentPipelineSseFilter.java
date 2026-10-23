package com.stonewu.fusion.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Set;

/**
 * SSE 由运行终态或客户端断开结束，不受 Servlet 默认异步超时或代理缓冲影响。
 */
@Component
public class AgentPipelineSseFilter extends OncePerRequestFilter {
    private static final Set<String> PATHS = Set.of(
            "/api/ai/pipeline/run", "/api/ai/pipeline/continue", "/api/ai/pipeline/reconnect");

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        return !PATHS.contains(request.getServletPath());
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        response.setHeader("Cache-Control", "no-cache, no-store, no-transform");
        response.setHeader("X-Accel-Buffering", "no");
        chain.doFilter(request, response);
        if (request.isAsyncStarted()) {
            request.getAsyncContext().setTimeout(0);
        }
    }
}
