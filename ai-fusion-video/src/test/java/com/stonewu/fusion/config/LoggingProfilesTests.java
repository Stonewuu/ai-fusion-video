package com.stonewu.fusion.config;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.stonewu.fusion.controller.system.AuthController;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.context.annotation.Configuration;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/** 启动日志系统，验证本机诊断日志及 Docker 中管理员找回密码日志的可见性。 */
class LoggingProfilesTests {
    @Configuration(proxyBeanMethods = false)
    static class Probe {
    }

    @ParameterizedTest
    @ValueSource(strings = {"local", "docker"})
    void profilesFilterActualLogEvents(String profile) {
        var root = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        var business = (Logger) LoggerFactory.getLogger("com.stonewu.fusion");
        var ai = (Logger) LoggerFactory.getLogger("org.springframework.ai");
        var recovery = (Logger) LoggerFactory.getLogger(AuthController.class);
        Level originalRoot = root.getLevel();
        Level originalBusiness = business.getLevel();
        Level originalAi = ai.getLevel();
        Level originalRecovery = recovery.getLevel();
        var application = new SpringApplication(Probe.class);
        application.setWebApplicationType(WebApplicationType.NONE);
        application.setLogStartupInfo(false);
        application.setRegisterShutdownHook(false);
        try (var context = application.run("--spring.profiles.active=" + profile)) {
            assertThat(events("com.stonewu.fusion.logging-probe"))
                    .containsExactlyElementsOf(profile.equals("docker")
                            ? List.of(Level.ERROR) : List.of(Level.DEBUG, Level.INFO, Level.WARN, Level.ERROR));
            if (profile.equals("docker")) {
                assertThat(events("org.example.logging-probe")).containsExactly(Level.ERROR);
                assertThat(events("org.springframework.ai.logging-probe")).containsExactly(Level.ERROR);
                assertThat(events(AuthController.class.getName())).containsExactly(Level.WARN, Level.ERROR);
            }
        } finally {
            root.setLevel(originalRoot);
            business.setLevel(originalBusiness);
            ai.setLevel(originalAi);
            recovery.setLevel(originalRecovery);
        }
    }

    private List<Level> events(String name) {
        var logger = (Logger) LoggerFactory.getLogger(name);
        var output = new ListAppender<ILoggingEvent>();
        output.start();
        logger.addAppender(output);
        try {
            logger.debug("debug-probe");
            logger.info("info-probe");
            logger.warn("warn-probe");
            logger.error("error-probe");
            return output.list.stream().map(ILoggingEvent::getLevel).toList();
        } finally {
            logger.detachAppender(output);
            output.stop();
        }
    }
}
