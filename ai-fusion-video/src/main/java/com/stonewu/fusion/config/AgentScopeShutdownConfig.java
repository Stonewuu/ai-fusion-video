package com.stonewu.fusion.config;

import io.agentscope.core.shutdown.GracefulShutdownConfig;
import io.agentscope.core.shutdown.GracefulShutdownManager;
import io.agentscope.core.shutdown.PartialReasoningPolicy;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

import java.time.Duration;

/**
 * 本地开发时覆盖 AgentScope 默认的无限 graceful shutdown 等待时间。
 */
@Configuration
@Profile("local")
@Slf4j
public class AgentScopeShutdownConfig {

    @Value("${fusion.agentscope.shutdown-timeout:3s}")
    private Duration shutdownTimeout;

    @PostConstruct
    public void configureGracefulShutdown() {
        GracefulShutdownManager.getInstance().setConfig(
                new GracefulShutdownConfig(shutdownTimeout, PartialReasoningPolicy.SAVE));
        log.info("[AgentScope] 本地 graceful shutdown 超时已设置为 {}", shutdownTimeout);
    }
}