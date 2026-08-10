package com.stonewu.fusion.infrastructure.queue;

import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * 通用 Redis 任务队列（多实例并发安全）
 * <p>
 * 使用 Redis 原子操作确保多实例部署时的并发安全：
 * - LPOP/RPUSH：原子取/放任务
 * - INCR/DECR：原子控制并发计数
 */
@Component
@Slf4j
public class RedisTaskQueue {

    private static final String KEY_PREFIX = "fv:taskqueue:";
    private static final String QUEUE_REGISTRY_KEY = KEY_PREFIX + "registered_queues";
    private static final int DEFAULT_MAX_CONCURRENT = 1;

    @Resource
    private StringRedisTemplate stringRedisTemplate;

    /**
     * 将任务放入队列
     */
    public void push(String queueName, String taskData) {
        String queueKey = getQueueKey(queueName);
        stringRedisTemplate.opsForList().rightPush(queueKey, taskData);
        registerQueue(queueName);
        log.debug("[push] 任务入队，queue:{}, task:{}", queueName, taskData);
    }

    /** Remove one pending occurrence without affecting a running lease. */
    public boolean remove(String queueName, String taskData) {
        Long removed = stringRedisTemplate.opsForList()
                .remove(getQueueKey(queueName), 1, taskData);
        cleanupQueueRegistration(queueName);
        return removed != null && removed > 0;
    }

    /**
     * 尝试获取执行许可并取出任务
     */
    public String acquireAndPop(String queueName, int timeoutSeconds) {
        String concurrentKey = getConcurrentKey(queueName);
        String queueKey = getQueueKey(queueName);
        int maxConcurrent = getMaxConcurrent(queueName);

        Long current = stringRedisTemplate.opsForValue().increment(concurrentKey, 0);
        if (current != null && current >= maxConcurrent) {
            return null;
        }

        String taskData;
        try {
            taskData = stringRedisTemplate.opsForList().leftPop(queueKey, timeoutSeconds, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.error("[acquireAndPop] 取任务异常，queue:{}", queueName, e);
            return null;
        }

        if (taskData == null) {
            return null;
        }

        current = stringRedisTemplate.opsForValue().increment(concurrentKey);
        if (current == null) {
            stringRedisTemplate.opsForList().leftPush(queueKey, taskData);
            return null;
        }

        stringRedisTemplate.expire(concurrentKey, 10, TimeUnit.MINUTES);

        if (current > maxConcurrent) {
            stringRedisTemplate.opsForValue().decrement(concurrentKey);
            stringRedisTemplate.opsForList().leftPush(queueKey, taskData);
            return null;
        }

        log.info("[acquireAndPop] 获取任务，queue:{}, task:{}, concurrent:{}/{}",
                queueName, taskData, current, maxConcurrent);
        return taskData;
    }

    /**
     * 阻塞式获取执行许可（不涉及队列，仅控制并发数）
     * <p>
     * 适用于 AI Agent 工具等需要同步等待结果的场景。
     * 与 {@link #acquireAndPop} 共享同一个并发计数器，
     * 确保两条路径（Agent 生图 + Consumer 异步生图）的总并发不超限。
     *
     * @param queueName    队列名（用于定位并发计数器 key）
     * @param timeoutMs    最大等待时间（毫秒）
     * @param pollIntervalMs 轮询间隔（毫秒）
     * @return 是否成功获取许可
     * @throws InterruptedException 如果等待过程中线程被中断
     */
    public boolean acquire(String queueName, long timeoutMs, long pollIntervalMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        String concurrentKey = getConcurrentKey(queueName);

        while (System.currentTimeMillis() < deadline) {
            int maxConcurrent = getMaxConcurrent(queueName);

            // 先递增，再检查是否超限（原子操作保证并发安全）
            Long current = stringRedisTemplate.opsForValue().increment(concurrentKey);
            if (current != null && current <= maxConcurrent) {
                stringRedisTemplate.expire(concurrentKey, 10, TimeUnit.MINUTES);
                log.info("[acquire] 获取许可成功，queue:{}, concurrent:{}/{}",
                        queueName, current, maxConcurrent);
                return true;
            }

            // 超限，撤销递增
            if (current != null) {
                stringRedisTemplate.opsForValue().decrement(concurrentKey);
            }

            // 等待后重试
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) break;
            Thread.sleep(Math.min(pollIntervalMs, remaining));
        }

        log.warn("[acquire] 获取许可超时，queue:{}, waited:{}ms", queueName, timeoutMs);
        return false;
    }

    /**
     * 阻塞式获取执行许可（默认 2 秒轮询间隔）
     */
    public boolean acquire(String queueName, long timeoutMs) throws InterruptedException {
        return acquire(queueName, timeoutMs, 2000);
    }

    /**
     * 释放执行许可
     */
    public void release(String queueName) {
        String concurrentKey = getConcurrentKey(queueName);
        String value = stringRedisTemplate.opsForValue().get(concurrentKey);
        int currentValue = 0;
        if (value != null) {
            try {
                currentValue = Integer.parseInt(value);
            } catch (NumberFormatException e) {
                stringRedisTemplate.opsForValue().set(concurrentKey, "0");
                return;
            }
        }
        if (currentValue <= 0) {
            return;
        }
        Long afterDecr = stringRedisTemplate.opsForValue().decrement(concurrentKey);
        if (afterDecr != null && afterDecr < 0) {
            stringRedisTemplate.opsForValue().set(concurrentKey, "0");
        }
        cleanupQueueRegistration(queueName);
    }

    public int getConcurrentCount(String queueName) {
        String value = stringRedisTemplate.opsForValue().get(getConcurrentKey(queueName));
        if (value != null) {
            try {
                return Math.max(0, Integer.parseInt(value));
            } catch (NumberFormatException e) {
                return 0;
            }
        }
        return 0;
    }

    public int getQueueLength(String queueName) {
        Long size = stringRedisTemplate.opsForList().size(getQueueKey(queueName));
        return size != null ? size.intValue() : 0;
    }

    public int getMaxConcurrent(String queueName) {
        String value = stringRedisTemplate.opsForValue().get(getMaxConcurrentKey(queueName));
        if (value != null) {
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                return DEFAULT_MAX_CONCURRENT;
            }
        }
        return DEFAULT_MAX_CONCURRENT;
    }

    public void setMaxConcurrent(String queueName, int maxConcurrent) {
        stringRedisTemplate.opsForValue().set(getMaxConcurrentKey(queueName), String.valueOf(maxConcurrent));
        registerQueue(queueName);
    }

    public Set<String> listRegisteredQueuesByPrefix(String queuePrefix) {
        Set<String> queueNames = stringRedisTemplate.opsForSet().members(QUEUE_REGISTRY_KEY);
        if (queueNames == null || queueNames.isEmpty()) {
            return Collections.emptySet();
        }

        Set<String> matchedQueueNames = new LinkedHashSet<>();
        for (String queueName : queueNames) {
            if (queueName != null && queueName.startsWith(queuePrefix)) {
                matchedQueueNames.add(queueName);
            }
        }
        return matchedQueueNames;
    }

    /**
     * 标记任务开始执行
     */
    public void markRunning(String queueName, String taskId, int timeoutMinutes) {
        String runningKey = getRunningKey(queueName, taskId);
        stringRedisTemplate.opsForValue().set(runningKey, "1", timeoutMinutes, TimeUnit.MINUTES);
    }

    /**
     * 标记任务完成
     */
    public void markComplete(String queueName, String taskId) {
        stringRedisTemplate.delete(getRunningKey(queueName, taskId));
    }

    /**
     * 续约任务租约
     */
    public boolean renewLease(String queueName, String taskId, int timeoutMinutes) {
        String runningKey = getRunningKey(queueName, taskId);
        Boolean result = stringRedisTemplate.expire(runningKey, timeoutMinutes, TimeUnit.MINUTES);
        return Boolean.TRUE.equals(result);
    }

    private String getQueueKey(String queueName) {
        return KEY_PREFIX + queueName + ":queue";
    }

    private String getConcurrentKey(String queueName) {
        return KEY_PREFIX + queueName + ":concurrent";
    }

    private String getMaxConcurrentKey(String queueName) {
        return KEY_PREFIX + queueName + ":max_concurrent";
    }

    private String getRunningKey(String queueName, String taskId) {
        return KEY_PREFIX + queueName + ":running:" + taskId;
    }

    private void registerQueue(String queueName) {
        stringRedisTemplate.opsForSet().add(QUEUE_REGISTRY_KEY, queueName);
    }

    private void cleanupQueueRegistration(String queueName) {
        if (getQueueLength(queueName) > 0 || getConcurrentCount(queueName) > 0) {
            return;
        }
        stringRedisTemplate.opsForSet().remove(QUEUE_REGISTRY_KEY, queueName);
    }
}
