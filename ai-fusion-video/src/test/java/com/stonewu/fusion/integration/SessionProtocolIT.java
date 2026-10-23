package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.mapper.system.UserMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;

/** 使用真实登录、刷新和重置 HTTP 请求证明协议结果；不模拟认证、Redis 或数据库。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class SessionProtocolIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "session-protocol-" + UUID.randomUUID()).toAbsolutePath();
    static final String PASSWORD = "test-original-password";

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry p) {
        p.add("spring.datasource.url", MYSQL::getJdbcUrl);
        p.add("spring.datasource.username", MYSQL::getUsername);
        p.add("spring.datasource.password", MYSQL::getPassword);
        p.add("spring.data.redis.host", REDIS::getHost);
        p.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        p.add("spring.data.redis.password", () -> "");
        p.add("app.storage.local-base-path", FILES::toString);
        p.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }

    @Autowired TestRestTemplate http;
    @Autowired UserMapper users;
    @Autowired PasswordEncoder passwords;
    @Autowired StringRedisTemplate redis;

    @Test
    void concurrentRefreshCanCreateOnlyOneUsableReplacementSession() {
        User user = user();
        var initial = ok(login(user, PASSWORD));
        var results = parallel(8, ignored -> http.postForEntity("/api/auth/refresh",
                Map.of("refreshToken", initial.path("refreshToken").asText()), JsonNode.class));
        var successful = results.stream().filter(this::successful).map(response -> response.getBody().path("data"))
                .toList();
        long usable = successful.stream().filter(pair -> successful(info(pair.path("accessToken").asText()))).count();
        assertThat(usable).as("The same refresh credential created %s usable sessions from %s successful responses",
                usable, successful.size()).isEqualTo(1);
        assertThat(info(initial.path("accessToken").asText()).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        var winner = successful.getFirst();
        assertThat(ok(info(winner.path("accessToken").asText())).path("id").asLong()).isEqualTo(user.getId());
        var replay = http.postForEntity("/api/auth/refresh",
                Map.of("refreshToken", initial.path("refreshToken").asText()), JsonNode.class);
        assertThat(replay.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void concurrentPasswordResetConsumesItsCredentialOnlyOnce() {
        User user = user();
        String token = resetCredential(user);
        var results = parallel(6, index -> reset(token, "test-replacement-" + index));
        long successes = results.stream().filter(this::successful).count();
        long acceptedPasswords = IntStream.range(0, 6).filter(index ->
                successful(login(user, "test-replacement-" + index))).count();
        assertThat(successes).as("The same reset credential changed the password successfully %s times; "
                + "final passwords accepted=%s", successes, acceptedPasswords).isEqualTo(1);
        assertThat(acceptedPasswords).isEqualTo(1);
        assertThat(successful(reset(token, "test-replayed-password"))).isFalse();
    }

    @Test
    void resetCannotAcceptAPasswordThatTheLoginProtocolWillReject() {
        for (String invalid : List.of("short", "x".repeat(33))) {
            User user = user();
            String token = resetCredential(user);
            var response = reset(token, invalid);
            var newLogin = login(user, invalid);
            assertThat(response.getStatusCode()).as("Reset accepted=%s while login with the new password returned %s",
                    successful(response), newLogin.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
            ok(login(user, PASSWORD));
            ok(reset(token, "test-valid-new-password"));
            ok(login(user, "test-valid-new-password"));
        }
    }

    private User user() {
        User user = User.builder().username("session-" + UUID.randomUUID()).nickname("会话协议测试")
                .password(passwords.encode(PASSWORD)).status(1).build();
        users.insert(user);
        return user;
    }

    private String resetCredential(User user) {
        // 夹具仅替代邮件投递，建立与申请找回入口相同的已签发、未消费凭据。
        String token = UUID.randomUUID().toString().replace("-", "");
        redis.opsForValue().set("auth:reset-token:email:" + token, user.getId().toString(), Duration.ofHours(24));
        return token;
    }

    private ResponseEntity<JsonNode> login(User user, String password) {
        return http.postForEntity("/api/auth/login", Map.of("username", user.getUsername(), "password", password),
                JsonNode.class);
    }

    private ResponseEntity<JsonNode> reset(String token, String password) {
        return http.postForEntity("/api/auth/reset-password/submit", Map.of("token", token, "type", "email",
                "newPassword", password, "confirmPassword", password), JsonNode.class);
    }

    private ResponseEntity<JsonNode> info(String token) {
        var headers = new HttpHeaders();
        headers.setBearerAuth(token);
        headers.setContentType(MediaType.APPLICATION_JSON);
        return http.exchange("/api/auth/user-info", HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class);
    }

    private List<ResponseEntity<JsonNode>> parallel(int count, IntFunction<ResponseEntity<JsonNode>> request) {
        var start = new CountDownLatch(1);
        var calls = IntStream.range(0, count).mapToObj(index -> CompletableFuture.supplyAsync(() -> {
            try {
                start.await();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(interrupted);
            }
            return request.apply(index);
        })).toList();
        start.countDown();
        return calls.stream().map(CompletableFuture::join).toList();
    }

    private boolean successful(ResponseEntity<JsonNode> response) {
        return response.getStatusCode() == HttpStatus.OK && response.getBody() != null
                && response.getBody().path("code").asInt(-1) == 0;
    }

    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(successful(response)).isTrue();
        return response.getBody().path("data");
    }
}
