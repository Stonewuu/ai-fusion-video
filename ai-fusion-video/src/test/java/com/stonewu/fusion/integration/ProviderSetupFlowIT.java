package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.ai.ProviderCatalogController;
import com.stonewu.fusion.controller.ai.vo.ApiConfigSaveReqVO;
import com.stonewu.fusion.controller.ai.vo.ProviderSetupReqVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ChatModelFactory;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.catalog.MetasoModelDiscovery;
import com.stonewu.fusion.service.ai.catalog.ProviderCatalogService;
import com.stonewu.fusion.service.ai.catalog.ProviderDiscoveryService;
import com.stonewu.fusion.service.ai.catalog.ProviderModelCandidate;
import com.stonewu.fusion.service.ai.catalog.ProviderSetupService;
import com.stonewu.fusion.service.ai.catalog.ProviderSetupWriter;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.sun.net.httpserver.HttpServer;
import com.zaxxer.hikari.HikariDataSource;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.concurrent.ConcurrentMapCacheManager;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.dao.DataAccessResourceFailureException;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.transaction.IllegalTransactionStateException;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.reset;
import static org.mockito.ArgumentMatchers.argThat;

/** 真实 MySQL + Spring 事务/权限代理验证快速配置的原子性、幂等和已配置模型保护。 */
@Testcontainers
class ProviderSetupFlowIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("provider_setup").withUsername("afv").withPassword("afv-test");
    private static HikariDataSource source;
    private static AnnotationConfigApplicationContext context;
    private static HttpServer server;
    private static final AtomicInteger REQUESTS = new AtomicInteger();
    private ProviderSetupService setup;
    private ProviderDiscoveryService discovery;
    private ApiConfigMapper configs;
    private AiModelMapper models;

    @Configuration
    @EnableTransactionManagement
    @EnableCaching
    @EnableMethodSecurity
    static class Infrastructure {
        @Bean
        static BeanPostProcessor databaseFailureInjection() {
            return new BeanPostProcessor() {
                @Override
                public Object postProcessAfterInitialization(Object bean, String name) {
                    return bean instanceof AiModelMapper ? spy(bean) : bean;
                }
            };
        }
    }

    @BeforeAll
    static void start() throws Exception {
        source = new HikariDataSource();
        source.setJdbcUrl(MYSQL.getJdbcUrl());
        source.setUsername(MYSQL.getUsername());
        source.setPassword(MYSQL.getPassword());
        source.setMaximumPoolSize(3);
        var flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        context = new AnnotationConfigApplicationContext();
        context.register(Infrastructure.class);
        context.registerBean(ObjectMapper.class, () -> new ObjectMapper());
        context.registerBean(PlatformTransactionManager.class, () -> new DataSourceTransactionManager(source));
        context.registerBean(CacheManager.class, () -> new ConcurrentMapCacheManager());
        context.registerBean(ChatModelFactory.class, () -> mock(ChatModelFactory.class));
        context.registerBean(ComfyUiWorkflowService.class, () -> mock(ComfyUiWorkflowService.class));
        MybatisTestSupport.register(context, source, ApiConfigMapper.class, AiModelMapper.class);
        context.register(ProviderCatalogService.class, ModelPresetService.class, ApiConfigService.class,
                AiModelMetadataResolver.class, AiModelService.class, MetasoModelDiscovery.class,
                ProviderDiscoveryService.class, ProviderSetupWriter.class, ProviderSetupService.class,
                ProviderCatalogController.class);
        context.refresh();
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/api/minimax/v2/credits/balance", exchange -> {
            REQUESTS.incrementAndGet();
            byte[] body = "{\"balance\":20}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
    }

    @AfterAll
    static void stop() {
        if (server != null) {
            server.stop(0);
        }
        if (context != null) {
            context.close();
        }
        if (source != null) {
            source.close();
        }
    }

    @BeforeEach
    void services() {
        login("ADMIN");
        setup = context.getBean(ProviderSetupService.class);
        discovery = context.getBean(ProviderDiscoveryService.class);
        configs = context.getBean(ApiConfigMapper.class);
        models = context.getBean(AiModelMapper.class);
        REQUESTS.set(0);
    }

    @AfterEach
    void logout() {
        SecurityContextHolder.clearContext();
    }

    /** 重试不能创建第二份账号；追加不能覆盖原模型的名称、默认选择、并发和参数。 */
    @Test
    void previewIsReadOnlyAndRepeatedSetupPreservesUserSettings() {
        var request = request(List.of("MiniMax-H3"));
        long before = configs.selectCount(null);
        discovery.preview(request.configuration());
        assertThat(configs.selectCount(null)).isEqualTo(before);
        var created = setup.apply(request);
        assertThat(created.createdModels()).isEqualTo(1);
        int verified = REQUESTS.get();
        assertThat(setup.apply(request).apiConfigId()).isEqualTo(created.apiConfigId());
        assertThat(REQUESTS.get()).isEqualTo(verified);
        assertThat(configs.selectCount(null)).isEqualTo(before + 1);
        var existing = models.selectOne(new LambdaQueryWrapper<AiModel>()
                .eq(AiModel::getApiConfigId, created.apiConfigId()));
        existing.setName("我自己的显示名称");
        existing.setConfig("{\"fastMode\":true}");
        existing.setMaxConcurrency(2);
        existing.setDefaultModel(true);
        models.updateById(existing);
        request.configuration().setId(created.apiConfigId());
        var appended = setup.apply(new ProviderSetupReqVO(request.configuration(), UUID.randomUUID().toString(),
                List.of("MiniMax-H3", "seedance-2.0-mini")));
        assertThat(appended.createdModels()).isEqualTo(1);
        assertThat(appended.existingModels()).isEqualTo(1);
        var preserved = models.selectById(existing.getId());
        assertThat(preserved.getName()).isEqualTo("我自己的显示名称");
        assertThat(preserved.getConfig()).isEqualTo("{\"fastMode\":true}");
        assertThat(preserved.getMaxConcurrency()).isEqualTo(2);
        assertThat(preserved.getDefaultModel()).isTrue();
    }

    @Test
    void concurrentRetriesCreateOneConfigurationAndOneSetOfModels() throws Exception {
        var request = request(List.of("MiniMax-H3", "seedance-2.0"));
        long before = configs.selectCount(null);
        try (var workers = Executors.newFixedThreadPool(2)) {
            var first = workers.submit(() -> {
                login("ADMIN");
                try {
                    return setup.apply(request);
                } finally {
                    SecurityContextHolder.clearContext();
                }
            });
            var second = workers.submit(() -> {
                login("ADMIN");
                try {
                    return setup.apply(request);
                } finally {
                    SecurityContextHolder.clearContext();
                }
            });
            var result = first.get();
            assertThat(second.get().apiConfigId()).isEqualTo(result.apiConfigId());
            assertThat(configs.selectCount(null)).isEqualTo(before + 1);
            assertThat(models.selectCount(new LambdaQueryWrapper<AiModel>()
                    .eq(AiModel::getApiConfigId, result.apiConfigId()))).isEqualTo(2);
        }
        var changed = new ProviderSetupReqVO(request.configuration(), request.requestId(), List.of("MiniMax-H3"));
        assertThatThrownBy(() -> setup.apply(changed)).hasMessageContaining("请求已经完成");
    }

    @Test
    void databaseFailureRollsBackConnectionAndEarlierModels() {
        var config = discovery.resolveConfiguration(request(List.of("MiniMax-H3")).configuration());
        long beforeConfigs = configs.selectCount(null);
        long beforeModels = models.selectCount(null);
        var valid = new ProviderModelCandidate("MiniMax-H3", "H3", 3, "metaso_video", "metaso:MiniMax-H3", "{}", null);
        var second = new ProviderModelCandidate("seedance-2.0-mini", "Seedance Mini", 3, "metaso_video",
                "metaso:seedance-2.0-mini", "{}", null);
        doThrow(new DataAccessResourceFailureException("simulated database write failure"))
                .when(models).insert(argThat((AiModel model) -> "seedance-2.0-mini".equals(model.getCode())));
        try {
            assertThatThrownBy(() -> context.getBean(ProviderSetupWriter.class).save(config, List.of(valid, second),
                    UUID.randomUUID().toString(), "test-hash")).isInstanceOf(DataAccessResourceFailureException.class);
        } finally {
            reset(models);
        }
        assertThat(configs.selectCount(null)).isEqualTo(beforeConfigs);
        assertThat(models.selectCount(null)).isEqualTo(beforeModels);
    }

    @Test
    void nonAdminsAndOuterTransactionsCannotStartExternalDiscovery() {
        var request = request(List.of("MiniMax-H3"));
        login("USER");
        var controller = context.getBean(ProviderCatalogController.class);
        assertThatThrownBy(() -> controller.preview(request.configuration())).isInstanceOf(AccessDeniedException.class);
        assertThatThrownBy(() -> controller.setup(request)).isInstanceOf(AccessDeniedException.class);
        login("ADMIN");
        var transaction = new TransactionTemplate(context.getBean(PlatformTransactionManager.class));
        assertThatThrownBy(() -> transaction.execute(status -> setup.apply(request)))
                .isInstanceOf(IllegalTransactionStateException.class);
        assertThat(REQUESTS.get()).isZero();
    }

    private ProviderSetupReqVO request(List<String> ids) {
        var config = new ApiConfigSaveReqVO();
        config.setName("秘塔测试 " + UUID.randomUUID());
        config.setProviderId("metaso");
        config.setApiKey("local-only-key");
        config.setApiUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/api/minimax");
        return new ProviderSetupReqVO(config, UUID.randomUUID().toString(), ids);
    }

    private static void login(String role) {
        var principal = new SecurityUserDetails(1001L, "provider-test", "unused", 1, null,
                List.of(new SimpleGrantedAuthority("ROLE_" + role)));
        SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken(
                principal, null, principal.getAuthorities()));
    }
}
