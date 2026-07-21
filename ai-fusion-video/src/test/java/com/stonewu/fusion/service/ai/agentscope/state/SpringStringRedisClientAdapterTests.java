package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.common.BusinessException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.core.Cursor;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.RedisCallback;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.data.redis.core.SetOperations;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class SpringStringRedisClientAdapterTests {

    private final StringRedisTemplate redisTemplate = mock(StringRedisTemplate.class);
    private final ValueOperations<String, String> valueOperations = mock(ValueOperations.class);
    private final ListOperations<String, String> listOperations = mock(ListOperations.class);
    private final SetOperations<String, String> setOperations = mock(SetOperations.class);

    private SpringStringRedisClientAdapter adapter;

    @BeforeEach
    void setUp() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(redisTemplate.opsForList()).thenReturn(listOperations);
        when(redisTemplate.opsForSet()).thenReturn(setOperations);
        adapter = new SpringStringRedisClientAdapter(redisTemplate, 4);
    }

    @Test
    void delegatesEveryStringListSetDeleteAndExistsOperation() {
        when(valueOperations.get("state:key")).thenReturn("value");
        when(listOperations.range("state:list", 0L, -1L)).thenReturn(List.of("first", "second"));
        when(listOperations.size("state:list")).thenReturn(2L);
        when(setOperations.members("state:set")).thenReturn(Set.of("member"));
        when(setOperations.size("state:set")).thenReturn(1L);
        when(redisTemplate.hasKey("state:key")).thenReturn(true);

        adapter.set("state:key", "value");
        assertThat(adapter.get("state:key")).isEqualTo("value");
        adapter.rightPushList("state:list", "first");
        assertThat(adapter.rangeList("state:list", 0, -1)).containsExactly("first", "second");
        assertThat(adapter.getListLength("state:list")).isEqualTo(2L);
        adapter.addToSet("state:set", "member");
        assertThat(adapter.getSetMembers("state:set")).containsExactly("member");
        assertThat(adapter.getSetSize("state:set")).isEqualTo(1L);
        assertThat(adapter.keyExists("state:key")).isTrue();
        adapter.deleteKeys("state:key", null, "state:list");

        verify(valueOperations).set("state:key", "value");
        verify(valueOperations).get("state:key");
        verify(listOperations).rightPush("state:list", "first");
        verify(redisTemplate).delete(List.of("state:key", "state:list"));
    }

    @Test
    void normalizesNullRedisCollectionsAndIgnoresEmptyDeletes() {
        when(listOperations.range("missing:list", 0L, -1L)).thenReturn(null);
        when(listOperations.size("missing:list")).thenReturn(null);
        when(setOperations.members("missing:set")).thenReturn(null);
        when(setOperations.size("missing:set")).thenReturn(null);

        assertThat(adapter.rangeList("missing:list", 0, -1)).isEmpty();
        assertThat(adapter.getListLength("missing:list")).isZero();
        assertThat(adapter.getSetMembers("missing:set")).isEmpty();
        assertThat(adapter.getSetSize("missing:set")).isZero();
        adapter.deleteKeys();
        adapter.deleteKeys((String[]) null);
        adapter.deleteKeys(null, null);

        verify(redisTemplate, never()).delete(anyList());
    }

    @Test
    @SuppressWarnings("unchecked")
    void scansWithCursorAndClosesIt() {
        RedisConnection connection = mock(RedisConnection.class);
        Cursor<byte[]> cursor = mock(Cursor.class);
        when(redisTemplate.execute(any(RedisCallback.class))).thenAnswer(invocation -> {
            RedisCallback<Set<String>> callback = invocation.getArgument(0);
            return callback.doInRedis(connection);
        });
        when(connection.scan(any(ScanOptions.class))).thenReturn(cursor);
        when(cursor.hasNext()).thenReturn(true, true, false);
        when(cursor.next()).thenReturn(
                "state:a".getBytes(StandardCharsets.UTF_8),
                "state:b".getBytes(StandardCharsets.UTF_8));

        assertThat(adapter.findKeysByPattern("state:*")).containsExactlyInAnyOrder("state:a", "state:b");

        verify(connection).scan(any(ScanOptions.class));
        verify(cursor).close();
    }

    @Test
    void rejectsImmediatelyWhenBulkheadIsFullAndReleasesPermitAfterCompletion() throws Exception {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(valueOperations.get("held")).thenAnswer(ignored -> {
            entered.countDown();
            if (!release.await(5, TimeUnit.SECONDS)) {
                throw new IllegalStateException("test operation was not released");
            }
            return "held-value";
        });
        when(valueOperations.get("after")).thenReturn("after-value");
        adapter = new SpringStringRedisClientAdapter(redisTemplate, 1);

        try (ExecutorService executor = Executors.newSingleThreadExecutor()) {
            Future<String> held = executor.submit(() -> adapter.get("held"));
            assertThat(entered.await(5, TimeUnit.SECONDS)).isTrue();

            assertThatThrownBy(() -> adapter.get("rejected"))
                    .isInstanceOfSatisfying(BusinessException.class, failure -> {
                        assertThat(failure.getCode()).isEqualTo(503);
                        assertThat(failure.getMessage()).isEqualTo("STATE_STORE_FAILED: bulkhead full");
                    });

            release.countDown();
            assertThat(held.get(5, TimeUnit.SECONDS)).isEqualTo("held-value");
            assertThat(adapter.get("after")).isEqualTo("after-value");
        } finally {
            release.countDown();
        }
    }

    @Test
    void releasesBulkheadPermitWhenRedisThrows() {
        when(valueOperations.get("broken")).thenThrow(new IllegalStateException("redis down"));
        when(valueOperations.get("after")).thenReturn("recovered");
        adapter = new SpringStringRedisClientAdapter(redisTemplate, 1);

        assertThatThrownBy(() -> adapter.get("broken"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("redis down");
        assertThat(adapter.get("after")).isEqualTo("recovered");
    }

    @Test
    void closeDoesNotTouchSpringOwnedRedisLifecycle() {
        StringRedisTemplate springOwned = mock(StringRedisTemplate.class);

        new SpringStringRedisClientAdapter(springOwned, 1).close();

        verifyNoInteractions(springOwned);
    }
}
