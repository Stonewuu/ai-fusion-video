package com.stonewu.fusion.support;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.MapPropertySource;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import static org.assertj.core.api.Assertions.*;

class LocalServiceGuardFactoryTests {
    @ParameterizedTest @ValueSource(booleans = {true, false})
    void rejectsPersistentServicesBeforeCreatingAnyConnectionBean(boolean mysql) {
        var connections = new AtomicInteger();
        try (var context = new AnnotationConfigApplicationContext()) {
            context.getEnvironment().getPropertySources().addFirst(new MapPropertySource("test", mysql
                    ? Map.of("spring.datasource.url", "jdbc:mysql://127.0.0.1:43306/ai_fusion_video")
                    : Map.of("spring.data.redis.port", "46379")));
            new LocalServiceGuardFactory().createContextCustomizer(getClass(), List.of()).customizeContext(context, null);
            context.registerBean("connection", String.class, () -> { connections.incrementAndGet(); return "connection"; });
            assertThatThrownBy(context::refresh).isInstanceOf(IllegalStateException.class).hasMessageContaining("测试禁止连接本地开发");
            assertThat(connections).hasValue(0);
        }
    }
}
