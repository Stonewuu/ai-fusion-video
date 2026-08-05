package com.stonewu.fusion.config;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.ZoneId;
import java.util.TimeZone;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.test.util.TestPropertyValues;
import org.springframework.context.support.GenericApplicationContext;

class ApplicationTimeZoneInitializerTests {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withInitializer(new ConfigDataApplicationContextInitializer())
            .withPropertyValues(
                    "APP_TIME_ZONE=Asia/Shanghai",
                    "MYSQL_SESSION_TIME_ZONE=+08:00");

    @Test
    void loadsApplicationAndMysqlSessionTimeZonesFromConfiguration() {
        contextRunner.run(context -> {
            assertThat(context.getEnvironment().getProperty("app.time-zone"))
                    .isEqualTo("Asia/Shanghai");
            assertThat(context.getEnvironment().getProperty("app.mysql.session-time-zone"))
                    .isEqualTo("+08:00");
            assertThat(context.getEnvironment().getProperty("spring.datasource.hikari.connection-init-sql"))
                    .isEqualTo("SET time_zone = '+08:00'");
        });
    }

    @Test
    void appliesConfiguredApplicationTimeZone() {
        TimeZone originalTimeZone = TimeZone.getDefault();
        GenericApplicationContext context = new GenericApplicationContext();
        TestPropertyValues.of("app.time-zone=Asia/Shanghai").applyTo(context);
        try {
            TimeZone.setDefault(TimeZone.getTimeZone("UTC"));

            new ApplicationTimeZoneInitializer().initialize(context);

            assertThat(ZoneId.systemDefault()).isEqualTo(ZoneId.of("Asia/Shanghai"));
        } finally {
            context.close();
            TimeZone.setDefault(originalTimeZone);
        }
    }

    @Test
    void rejectsInvalidApplicationTimeZone() {
        GenericApplicationContext context = new GenericApplicationContext();
        TestPropertyValues.of("app.time-zone=invalid-zone").applyTo(context);
        try {
            assertThatThrownBy(() -> new ApplicationTimeZoneInitializer().initialize(context))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("app.time-zone")
                    .hasMessageContaining("invalid-zone");
        } finally {
            context.close();
        }
    }
}
