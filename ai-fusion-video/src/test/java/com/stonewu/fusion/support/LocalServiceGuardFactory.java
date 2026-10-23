package com.stonewu.fusion.support;

import org.springframework.test.context.ContextConfigurationAttributes;
import org.springframework.test.context.ContextCustomizer;
import org.springframework.test.context.ContextCustomizerFactory;
import java.util.List;

/** 在任何数据库/Redis Bean 创建前拒绝仓库本地开发端口，测试必须显式配置隔离服务。 */
public final class LocalServiceGuardFactory implements ContextCustomizerFactory {
    private static final ContextCustomizer GUARD = (context, configuration) -> context.addBeanFactoryPostProcessor(factory -> {
        var environment = context.getEnvironment();
        String database = environment.getProperty("spring.datasource.url", "");
        if (database.matches("(?i)jdbc:mysql://[^/]*:43306(?:/.*)?"))
            throw new IllegalStateException("测试禁止连接本地开发 MySQL 43306，请配置 Testcontainers 数据源");
        if ("46379".equals(environment.getProperty("spring.data.redis.port")))
            throw new IllegalStateException("测试禁止连接本地开发 Redis 46379，请配置独立 Redis 容器");
    });

    @Override public ContextCustomizer createContextCustomizer(Class<?> testClass, List<ContextConfigurationAttributes> attributes) {
        return GUARD;
    }
}
