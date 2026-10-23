package com.stonewu.fusion.service.media.lifecycle;

import jakarta.annotation.PreDestroy;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.integration.redis.util.RedisLockRegistry;
import org.springframework.scheduling.concurrent.ThreadPoolTaskScheduler;

import java.time.Duration;

@Configuration(proxyBeanMethods = false)
public class MediaBlobLockConfiguration {
    // 不注册为通用 TaskScheduler，避免业务定时任务占用锁续租线程。
    private final ThreadPoolTaskScheduler renewal = new ThreadPoolTaskScheduler();

    public MediaBlobLockConfiguration() {
        renewal.setPoolSize(2);
        renewal.setThreadNamePrefix("media-blob-lock-");
        renewal.setRemoveOnCancelPolicy(true);
        renewal.initialize();
    }

    @Bean
    public RedisLockRegistry mediaBlobLockRegistry(RedisConnectionFactory connections) {
        var registry = new RedisLockRegistry(connections, "afv:media:blob", Duration.ofMinutes(5).toMillis());
        registry.setRenewalTaskScheduler(renewal);
        return registry;
    }

    @PreDestroy
    public void close() {
        renewal.shutdown();
    }
}
