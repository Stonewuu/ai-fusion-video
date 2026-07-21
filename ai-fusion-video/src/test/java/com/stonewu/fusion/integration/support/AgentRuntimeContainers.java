package com.stonewu.fusion.integration.support;

import org.springframework.test.context.DynamicPropertyRegistry;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.utility.DockerImageName;

public final class AgentRuntimeContainers {

    public static final int REDIS_PORT = 6379;
    public static final DockerImageName MYSQL_IMAGE =
            DockerImageName.parse("mysql:8.4.6");
    public static final DockerImageName REDIS_IMAGE =
            DockerImageName.parse("redis:7.4.5-alpine");

    private AgentRuntimeContainers() {
    }

    public static MySQLContainer<?> mysql() {
        return new MySQLContainer<>(MYSQL_IMAGE)
                .withDatabaseName("ai_fusion_video")
                .withUsername("afv")
                .withPassword("afv-test");
    }

    public static GenericContainer<?> redis() {
        return new GenericContainer<>(REDIS_IMAGE).withExposedPorts(REDIS_PORT);
    }

    public static void registerSpringProperties(
            DynamicPropertyRegistry properties,
            MySQLContainer<?> mysql,
            GenericContainer<?> redis) {
        properties.add("spring.datasource.url", mysql::getJdbcUrl);
        properties.add("spring.datasource.username", mysql::getUsername);
        properties.add("spring.datasource.password", mysql::getPassword);
        properties.add("spring.datasource.hikari.minimum-idle", () -> 0);
        properties.add("spring.data.redis.host", redis::getHost);
        properties.add("spring.data.redis.port", () -> redis.getMappedPort(REDIS_PORT));
        properties.add("fusion.agentscope.v2.state.mode", () -> "in-memory");
    }
}
