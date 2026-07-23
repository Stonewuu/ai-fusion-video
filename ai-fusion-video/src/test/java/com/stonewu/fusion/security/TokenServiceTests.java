package com.stonewu.fusion.security;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class TokenServiceTests {

    private final Map<String, String> redis = new HashMap<>();
    private TokenService tokenService;

    @BeforeEach
    void setUp() {
        StringRedisTemplate redisTemplate = mock(StringRedisTemplate.class);
        @SuppressWarnings("unchecked")
        ValueOperations<String, String> valueOperations = mock(ValueOperations.class);

        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get(anyString())).thenAnswer(invocation -> redis.get(invocation.getArgument(0)));
        doAnswer(invocation -> {
            redis.put(invocation.getArgument(0), invocation.getArgument(1));
            return null;
        }).when(valueOperations).set(anyString(), anyString(), any(Duration.class));
        when(redisTemplate.delete(anyString()))
                .thenAnswer(invocation -> redis.remove(invocation.getArgument(0)) != null);
        when(redisTemplate.hasKey(anyString()))
                .thenAnswer(invocation -> redis.containsKey(invocation.getArgument(0)));
        when(redisTemplate.getExpire(anyString(), eq(TimeUnit.SECONDS))).thenReturn(3600L);

        tokenService = new TokenService(redisTemplate, new ObjectMapper());
    }

    @Test
    void secondLoginKeepsFirstLoginSessionValid() {
        TokenService.TokenPair first = tokenService.createToken(7L, "stone", 3L);
        TokenService.TokenPair second = tokenService.createToken(7L, "stone", 3L);

        assertThat(tokenService.getAccessTokenSession(first.getAccessToken())).isNotNull();
        assertThat(tokenService.getAccessTokenSession(second.getAccessToken())).isNotNull();
        assertThat(tokenService.refreshAccessToken(first.getRefreshToken())).isNotNull();
        assertThat(tokenService.getAccessTokenSession(second.getAccessToken())).isNotNull();
    }

    @Test
    void refreshingOneLoginDoesNotInvalidateAnotherLogin() {
        TokenService.TokenPair first = tokenService.createToken(7L, "stone", 3L);
        TokenService.TokenPair second = tokenService.createToken(7L, "stone", 3L);

        TokenService.TokenPair refreshedFirst = tokenService.refreshAccessToken(first.getRefreshToken());

        assertThat(refreshedFirst).isNotNull();
        assertThat(tokenService.getAccessTokenSession(first.getAccessToken())).isNull();
        assertThat(tokenService.getAccessTokenSession(refreshedFirst.getAccessToken())).isNotNull();
        assertThat(tokenService.getAccessTokenSession(second.getAccessToken())).isNotNull();
        assertThat(tokenService.refreshAccessToken(second.getRefreshToken())).isNotNull();
    }

    @Test
    void logoutRemovesOnlyCurrentLoginSession() {
        TokenService.TokenPair first = tokenService.createToken(7L, "stone", 3L);
        TokenService.TokenPair second = tokenService.createToken(7L, "stone", 3L);

        tokenService.removeToken(first.getAccessToken());

        assertThat(tokenService.getAccessTokenSession(first.getAccessToken())).isNull();
        assertThat(tokenService.refreshAccessToken(first.getRefreshToken())).isNull();
        assertThat(tokenService.getAccessTokenSession(second.getAccessToken())).isNotNull();
        assertThat(tokenService.refreshAccessToken(second.getRefreshToken())).isNotNull();
    }
}
