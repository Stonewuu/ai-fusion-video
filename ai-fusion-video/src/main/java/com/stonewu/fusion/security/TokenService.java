package com.stonewu.fusion.security;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.UUID;

/**
 * 基于 Redis 的 Token 服务
 * <p>
 * 使用 UUID 作为 token，支持 access_token + refresh_token 双 token 机制
 * <ul>
 *   <li>access_token：短期有效（2小时），用于接口鉴权</li>
 *   <li>refresh_token：长期有效（7天），用于刷新 access_token</li>
 * </ul>
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class TokenService {

    private static final String ACCESS_TOKEN_PREFIX = "fusion:token:";
    private static final String REFRESH_TOKEN_PREFIX = "fusion:refresh_token:";
    private static final String LEGACY_USER_TOKEN_PREFIX = "fusion:user_token:";
    private static final String ACCESS_REFRESH_SUFFIX = ":refresh";
    private static final String REFRESH_ACCESS_SUFFIX = ":access";

    /** access_token 有效期：2 小时 */
    private static final long ACCESS_TOKEN_EXPIRE_HOURS = 2;
    /** refresh_token 有效期：7 天 */
    private static final long REFRESH_TOKEN_EXPIRE_DAYS = 7;

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TokenSession {
        private Long userId;
        private String username;
        private Long currentTeamId;
    }

    /**
     * 令牌对，包含 access_token 和 refresh_token
     */
    @Data
    @AllArgsConstructor
    public static class TokenPair {
        /** 访问令牌 */
        private String accessToken;
        /** 刷新令牌 */
        private String refreshToken;
        /** access_token 有效期（秒） */
        private long expiresIn;
    }

    /**
     * 创建令牌对（登录/注册时调用）
     *
     * @param userId   用户ID
     * @param username 用户名
     * @return 包含 access_token 和 refresh_token 的令牌对
     */
    public TokenPair createToken(Long userId, String username, Long currentTeamId) {
        String userValue = serializeSession(new TokenSession(userId, username, currentTeamId));

        // 生成 access_token
        String accessToken = generateUUID();
        redisTemplate.opsForValue().set(
                ACCESS_TOKEN_PREFIX + accessToken,
                userValue,
                Duration.ofHours(ACCESS_TOKEN_EXPIRE_HOURS)
        );

        // 生成 refresh_token
        String refreshToken = generateUUID();
        redisTemplate.opsForValue().set(
                REFRESH_TOKEN_PREFIX + refreshToken,
                userValue,
                Duration.ofDays(REFRESH_TOKEN_EXPIRE_DAYS)
        );

        // 记录 access_token 对应的 refresh_token（用于登出时一起清除）
        redisTemplate.opsForValue().set(
                accessRefreshKey(accessToken),
                refreshToken,
                Duration.ofDays(REFRESH_TOKEN_EXPIRE_DAYS)
        );

        // 记录 refresh_token 对应的 access_token，刷新时只轮换当前登录会话
        redisTemplate.opsForValue().set(
                refreshAccessKey(refreshToken),
                accessToken,
                Duration.ofDays(REFRESH_TOKEN_EXPIRE_DAYS)
        );

        return new TokenPair(accessToken, refreshToken, ACCESS_TOKEN_EXPIRE_HOURS * 3600);
    }

    /**
     * 使用 refresh_token 刷新 access_token
     *
     * @param refreshToken 刷新令牌
     * @return 新的令牌对；如果 refresh_token 无效则返回 null
     */
    public TokenPair refreshAccessToken(String refreshToken) {
        TokenSession session = getRefreshTokenSession(refreshToken);
        if (session == null) {
            return null;
        }

        Long userId = session.getUserId();
        String oldAccessToken = findAccessTokenForRefresh(refreshToken, userId);
        if (oldAccessToken != null) {
            redisTemplate.delete(ACCESS_TOKEN_PREFIX + oldAccessToken);
            redisTemplate.delete(accessRefreshKey(oldAccessToken));
            removeLegacyUserTokenReference(userId, oldAccessToken);
        }

        // 只轮换当前 refresh_token 所属的会话，其他设备上的登录态保持有效
        redisTemplate.delete(REFRESH_TOKEN_PREFIX + refreshToken);
        redisTemplate.delete(refreshAccessKey(refreshToken));

        return createToken(userId, session.getUsername(), session.getCurrentTeamId());
    }

    /**
     * 根据 access_token 获取用户ID
     */
    public Long getUserIdFromToken(String token) {
        TokenSession session = getAccessTokenSession(token);
        return session != null ? session.getUserId() : null;
    }

    /**
     * 根据 access_token 获取用户名
     */
    public String getUsernameFromToken(String token) {
        TokenSession session = getAccessTokenSession(token);
        return session != null ? session.getUsername() : null;
    }

    /**
     * 根据 access_token 获取当前团队 ID
     */
    public Long getCurrentTeamIdFromToken(String token) {
        TokenSession session = getAccessTokenSession(token);
        return session != null ? session.getCurrentTeamId() : null;
    }

    /**
     * 读取 access_token 对应的认证会话
     */
    public TokenSession getAccessTokenSession(String token) {
        return deserializeSession(redisTemplate.opsForValue().get(ACCESS_TOKEN_PREFIX + token));
    }

    /**
     * 更新当前 access_token / refresh_token 会话中的团队 ID
     */
    public boolean updateCurrentTeam(String accessToken, Long expectedUserId, Long currentTeamId) {
        TokenSession session = getAccessTokenSession(accessToken);
        if (session == null || session.getUserId() == null) {
            return false;
        }
        if (expectedUserId != null && !expectedUserId.equals(session.getUserId())) {
            return false;
        }

        TokenSession updatedSession = new TokenSession(session.getUserId(), session.getUsername(), currentTeamId);
        String accessTokenKey = ACCESS_TOKEN_PREFIX + accessToken;
        writeSession(accessTokenKey, updatedSession, redisTemplate.getExpire(accessTokenKey, TimeUnit.SECONDS),
                Duration.ofHours(ACCESS_TOKEN_EXPIRE_HOURS));

        String refreshToken = redisTemplate.opsForValue().get(accessRefreshKey(accessToken));
        if (refreshToken != null) {
            String refreshTokenKey = REFRESH_TOKEN_PREFIX + refreshToken;
            writeSession(refreshTokenKey, updatedSession, redisTemplate.getExpire(refreshTokenKey, TimeUnit.SECONDS),
                    Duration.ofDays(REFRESH_TOKEN_EXPIRE_DAYS));
        }
        return true;
    }

    /**
     * 验证 access_token 是否有效
     */
    public boolean validateToken(String token) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(ACCESS_TOKEN_PREFIX + token));
    }

    /**
     * 删除令牌（登出时调用），同时清除 access_token 和 refresh_token
     */
    public void removeToken(String token) {
        // 清除关联的 refresh_token
        String refreshToken = redisTemplate.opsForValue().get(accessRefreshKey(token));
        if (refreshToken != null) {
            redisTemplate.delete(REFRESH_TOKEN_PREFIX + refreshToken);
            redisTemplate.delete(refreshAccessKey(refreshToken));
        }

        // 清除 access_token 及其关联键
        redisTemplate.delete(ACCESS_TOKEN_PREFIX + token);
        redisTemplate.delete(accessRefreshKey(token));
    }

    private String generateUUID() {
        return UUID.randomUUID().toString().replace("-", "");
    }

    private TokenSession getRefreshTokenSession(String refreshToken) {
        return deserializeSession(redisTemplate.opsForValue().get(REFRESH_TOKEN_PREFIX + refreshToken));
    }

    private String findAccessTokenForRefresh(String refreshToken, Long userId) {
        String accessToken = redisTemplate.opsForValue().get(refreshAccessKey(refreshToken));
        if (accessToken != null) {
            return accessToken;
        }

        // 兼容多端登录改造前签发的令牌对
        String legacyAccessToken = redisTemplate.opsForValue().get(LEGACY_USER_TOKEN_PREFIX + userId);
        if (legacyAccessToken == null) {
            return null;
        }
        String linkedRefreshToken = redisTemplate.opsForValue().get(accessRefreshKey(legacyAccessToken));
        return refreshToken.equals(linkedRefreshToken) ? legacyAccessToken : null;
    }

    private void removeLegacyUserTokenReference(Long userId, String accessToken) {
        String legacyKey = LEGACY_USER_TOKEN_PREFIX + userId;
        String legacyAccessToken = redisTemplate.opsForValue().get(legacyKey);
        if (accessToken.equals(legacyAccessToken)) {
            redisTemplate.delete(legacyKey);
        }
    }

    private String accessRefreshKey(String accessToken) {
        return ACCESS_TOKEN_PREFIX + accessToken + ACCESS_REFRESH_SUFFIX;
    }

    private String refreshAccessKey(String refreshToken) {
        return REFRESH_TOKEN_PREFIX + refreshToken + REFRESH_ACCESS_SUFFIX;
    }

    private String serializeSession(TokenSession session) {
        try {
            return objectMapper.writeValueAsString(session);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("序列化认证会话失败", e);
        }
    }

    private void writeSession(String key, TokenSession session, Long ttlSeconds, Duration fallbackTtl) {
        String rawValue = serializeSession(session);
        if (ttlSeconds != null && ttlSeconds > 0) {
            redisTemplate.opsForValue().set(key, rawValue, Duration.ofSeconds(ttlSeconds));
            return;
        }
        redisTemplate.opsForValue().set(key, rawValue, fallbackTtl);
    }

    private TokenSession deserializeSession(String rawValue) {
        if (rawValue == null || rawValue.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(rawValue, TokenSession.class);
        } catch (Exception ignored) {
            String[] parts = rawValue.split(":", 2);
            try {
                Long userId = Long.parseLong(parts[0]);
                String username = parts.length > 1 ? parts[1] : null;
                return new TokenSession(userId, username, null);
            } catch (Exception ex) {
                log.warn("解析认证会话失败: {}", rawValue, ex);
                return null;
            }
        }
    }
}
