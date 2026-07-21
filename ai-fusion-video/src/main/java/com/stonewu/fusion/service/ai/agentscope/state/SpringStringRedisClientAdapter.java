package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.common.BusinessException;
import io.agentscope.extensions.redis.state.RedisClientAdapter;
import org.springframework.data.redis.core.Cursor;
import org.springframework.data.redis.core.RedisCallback;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.function.Supplier;

public final class SpringStringRedisClientAdapter implements RedisClientAdapter {

    private static final long SCAN_COUNT = 1_000L;
    private static final String BULKHEAD_FULL_MESSAGE = "STATE_STORE_FAILED: bulkhead full";

    private final StringRedisTemplate redisTemplate;
    private final Semaphore bulkhead;

    public SpringStringRedisClientAdapter(StringRedisTemplate redisTemplate, int maxConcurrentOperations) {
        this.redisTemplate = Objects.requireNonNull(redisTemplate, "redisTemplate must not be null");
        if (maxConcurrentOperations <= 0) {
            throw new IllegalArgumentException("maxConcurrentOperations must be positive");
        }
        this.bulkhead = new Semaphore(maxConcurrentOperations, true);
    }

    @Override
    public void set(String key, String value) {
        execute(() -> redisTemplate.opsForValue().set(key, value));
    }

    @Override
    public String get(String key) {
        return execute(() -> redisTemplate.opsForValue().get(key));
    }

    @Override
    public void rightPushList(String key, String value) {
        execute(() -> redisTemplate.opsForList().rightPush(key, value));
    }

    @Override
    public List<String> rangeList(String key, long start, long end) {
        List<String> values = execute(() -> redisTemplate.opsForList().range(key, start, end));
        return values == null ? List.of() : List.copyOf(values);
    }

    @Override
    public long getListLength(String key) {
        Long size = execute(() -> redisTemplate.opsForList().size(key));
        return size == null ? 0L : size;
    }

    @Override
    public void deleteKeys(String... keys) {
        if (keys == null || keys.length == 0) {
            return;
        }
        List<String> safeKeys = Arrays.stream(keys)
                .filter(Objects::nonNull)
                .toList();
        if (!safeKeys.isEmpty()) {
            execute(() -> redisTemplate.delete(safeKeys));
        }
    }

    @Override
    public void addToSet(String key, String member) {
        execute(() -> redisTemplate.opsForSet().add(key, member));
    }

    @Override
    public Set<String> getSetMembers(String key) {
        Set<String> members = execute(() -> redisTemplate.opsForSet().members(key));
        return members == null ? Set.of() : Set.copyOf(members);
    }

    @Override
    public long getSetSize(String key) {
        Long size = execute(() -> redisTemplate.opsForSet().size(key));
        return size == null ? 0L : size;
    }

    @Override
    public boolean keyExists(String key) {
        return Boolean.TRUE.equals(execute(() -> redisTemplate.hasKey(key)));
    }

    @Override
    public Set<String> findKeysByPattern(String pattern) {
        Set<String> keys = execute(() -> redisTemplate.execute((RedisCallback<Set<String>>) connection -> {
            ScanOptions options = ScanOptions.scanOptions()
                    .match(pattern)
                    .count(SCAN_COUNT)
                    .build();
            Set<String> scannedKeys = new HashSet<>();
            try (Cursor<byte[]> cursor = connection.scan(options)) {
                while (cursor.hasNext()) {
                    scannedKeys.add(new String(cursor.next(), StandardCharsets.UTF_8));
                }
            }
            return scannedKeys;
        }));
        return keys == null ? Set.of() : Set.copyOf(keys);
    }

    @Override
    public void close() {
        // Spring owns the Redis connection lifecycle.
    }

    private void execute(Runnable action) {
        execute(() -> {
            action.run();
            return null;
        });
    }

    private <T> T execute(Supplier<T> action) {
        if (!bulkhead.tryAcquire()) {
            throw new BusinessException(503, BULKHEAD_FULL_MESSAGE);
        }
        try {
            return action.get();
        } finally {
            bulkhead.release();
        }
    }
}
