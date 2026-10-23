package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.core.config.GlobalConfig;
import com.baomidou.mybatisplus.extension.spring.MybatisSqlSessionFactoryBean;
import com.stonewu.fusion.config.MybatisPlusConfig;
import com.stonewu.fusion.config.MybatisPlusMetaHandler;
import org.mybatis.spring.SqlSessionTemplate;
import org.apache.ibatis.io.ResolverUtil;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.session.LocalCacheScope;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

import javax.sql.DataSource;
import java.util.Arrays;
import java.util.LinkedHashSet;

/**
 * 为隔离数据库的轻量集成上下文注册与应用一致的 Mapper。
 */
final class MybatisTestSupport {
    private MybatisTestSupport() {
    }

    static SqlSessionTemplate register(AnnotationConfigApplicationContext context, DataSource source,
                                       Class<?>... mappers)
            throws Exception {
        var factory = new MybatisSqlSessionFactoryBean();
        factory.setDataSource(source);
        var configuration = new MybatisConfiguration();
        configuration.setMapUnderscoreToCamelCase(true);
        configuration.setCallSettersOnNulls(true);
        configuration.setLocalCacheScope(LocalCacheScope.STATEMENT);
        factory.setConfiguration(configuration);
        factory.setGlobalConfig(new GlobalConfig().setMetaObjectHandler(new MybatisPlusMetaHandler())
                .setDbConfig(new GlobalConfig.DbConfig().setIdType(IdType.AUTO)
                        .setLogicDeleteField("deleted").setLogicDeleteValue("true").setLogicNotDeleteValue("false")));
        factory.setPlugins(new MybatisPlusConfig().mybatisPlusInterceptor());
        factory.setMapperLocations(
                new PathMatchingResourcePatternResolver().getResources("classpath*:mapper/**/*.xml"));
        var sessionFactory = factory.getObject();
        var session = new SqlSessionTemplate(sessionFactory);
        var types = new LinkedHashSet<Class<?>>(Arrays.asList(mappers));
        if (types.isEmpty()) {
            var resolver = new ResolverUtil<Object>();
            resolver.find(new ResolverUtil.IsA(Object.class), "com.stonewu.fusion.mapper");
            types.addAll(resolver.getClasses().stream()
                    .filter(type -> type.isInterface() && type.isAnnotationPresent(Mapper.class)).toList());
        }
        for (Class<?> mapper : types) {
            if (!configuration.hasMapper(mapper)) {
                configuration.addMapper(mapper);
            }
            registerMapper(context, session, mapper);
        }
        return session;
    }

    private static <T> void registerMapper(AnnotationConfigApplicationContext context, SqlSessionTemplate session,
                                           Class<T> type) {
        context.registerBean(type, () -> session.getMapper(type));
    }
}
