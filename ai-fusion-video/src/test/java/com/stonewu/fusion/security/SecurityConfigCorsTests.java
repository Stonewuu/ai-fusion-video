package com.stonewu.fusion.security;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.CorsProperties;
import org.junit.jupiter.api.Test;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

class SecurityConfigCorsTests {

    @Test
    void exposesOnlyConfiguredOriginPatterns() {
        CorsProperties properties = new CorsProperties();
        properties.setAllowedOriginPatterns(List.of(" https://app.example.com "));
        SecurityConfig securityConfig = new SecurityConfig(
                mock(TokenAuthenticationFilter.class), new ObjectMapper(), properties);

        UrlBasedCorsConfigurationSource source =
                (UrlBasedCorsConfigurationSource) securityConfig.corsConfigurationSource();
        CorsConfiguration configuration = source.getCorsConfigurations().get("/**");

        assertThat(configuration).isNotNull();
        assertThat(configuration.getAllowedOriginPatterns())
                .containsExactly("https://app.example.com");
        assertThat(configuration.getAllowCredentials()).isTrue();
    }

    @Test
    void rejectsWildcardOriginWhenCredentialsAreEnabled() {
        CorsProperties properties = new CorsProperties();
        properties.setAllowedOriginPatterns(List.of("*"));
        SecurityConfig securityConfig = new SecurityConfig(
                mock(TokenAuthenticationFilter.class), new ObjectMapper(), properties);

        assertThatThrownBy(securityConfig::corsConfigurationSource)
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("通配来源");
    }
}
