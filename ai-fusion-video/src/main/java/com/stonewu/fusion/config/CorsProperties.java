package com.stonewu.fusion.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 浏览器跨域访问配置。
 * <p>
 * 一体化部署和统一网关部署不需要配置；仅当前端直接访问不同域名的后端时配置允许来源。
 */
@Component
@ConfigurationProperties(prefix = "app.cors")
@Getter
@Setter
public class CorsProperties {

    private List<String> allowedOriginPatterns = new ArrayList<>();
}

