package com.stonewu.fusion.config;

import java.time.DateTimeException;
import java.time.ZoneId;
import java.util.TimeZone;

import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** 在 Spring Bean 初始化前应用配置的系统时区。 */
public class ApplicationTimeZoneInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    public static final String TIME_ZONE_PROPERTY = "app.time-zone";

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        String configuredTimeZone = applicationContext.getEnvironment().getRequiredProperty(TIME_ZONE_PROPERTY);
        try {
            ZoneId zoneId = ZoneId.of(configuredTimeZone);
            TimeZone.setDefault(TimeZone.getTimeZone(zoneId));
        } catch (DateTimeException exception) {
            throw new IllegalStateException(
                    "无效的应用时区配置 " + TIME_ZONE_PROPERTY + ": " + configuredTimeZone,
                    exception);
        }
    }
}
