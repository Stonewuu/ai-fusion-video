package com.stonewu.fusion.integration;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.zaxxer.hikari.HikariDataSource;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.assertj.core.api.Assertions.assertThat;

/** 验证升级保留自定义配置，并验证从内置提供商改回自定义时真正清除平台身份。 */
@Testcontainers
class ProviderCatalogPersistenceIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("provider_catalog").withUsername("afv").withPassword("afv-test");

    @Test
    void upgradesExistingConfigurationAndPersistsExplicitBindingChanges() throws Exception {
        try (var source = new HikariDataSource(); var context = new AnnotationConfigApplicationContext()) {
            source.setJdbcUrl(MYSQL.getJdbcUrl());
            source.setUsername(MYSQL.getUsername());
            source.setPassword(MYSQL.getPassword());
            Flyway.configure().dataSource(source).target("1.1.2.0.0").load().migrate();
            MybatisTestSupport.register(context, source, ApiConfigMapper.class);
            context.refresh();
            var mapper = context.getBean(ApiConfigMapper.class);
            var existing = ApiConfig.builder().name("已有自定义配置").platform("openai_compatible")
                    .apiUrl("https://example.com").apiKey("local-test-key").textProtocol("openai_compatible")
                    .build();
            mapper.insert(existing);

            var flyway = Flyway.configure().dataSource(source).load();
            flyway.migrate();
            flyway.validate();
            var reopened = mapper.selectById(existing.getId());
            assertThat(reopened.getProviderId()).isNull();
            assertThat(reopened.getApiUrl()).isEqualTo("https://example.com");
            assertThat(reopened.getApiKey()).isEqualTo("local-test-key");
            reopened.setProviderId("infistar");
            mapper.updateById(reopened);
            assertThat(mapper.selectById(existing.getId()).getProviderId()).isEqualTo("infistar");
            reopened.setProviderId(null);
            mapper.updateById(reopened);
            assertThat(mapper.selectById(existing.getId()).getProviderId()).isNull();
        }
    }
}
