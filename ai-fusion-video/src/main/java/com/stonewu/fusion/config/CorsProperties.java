package com.stonewu.fusion.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 浏览器跨域访问配置。默认允许所有来源；生产环境可以配置明确的来源白名单。
 */
@Component
@ConfigurationProperties(prefix = "app.cors")
@Getter
@Setter
public class CorsProperties {

    private List<String> allowedOriginPatterns = new ArrayList<>(List.of("*"));
}

