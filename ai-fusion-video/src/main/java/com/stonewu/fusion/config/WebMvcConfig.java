package com.stonewu.fusion.config;

import com.stonewu.fusion.service.storage.LegacyMediaRoot;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.AsyncTaskExecutor;
import org.springframework.web.servlet.config.annotation.AsyncSupportConfigurer;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Web MVC 配置
 * <p>
 * 将 /media/** HTTP 路径映射到本地磁盘存储目录，
 * 并将 /api/art-styles/** 映射到 classpath 中的默认画风参考图目录，
 * 使前端和后端都可以通过统一 URL 访问这些资源。
 */
@Configuration
@Slf4j
public class WebMvcConfig implements WebMvcConfigurer {

    private final LegacyMediaRoot legacyMediaRoot;
    private final AsyncTaskExecutor mvcAsyncExecutor;

    public WebMvcConfig(
            LegacyMediaRoot legacyMediaRoot,
            @Qualifier("mvcAsyncExecutor") AsyncTaskExecutor mvcAsyncExecutor) {
        this.legacyMediaRoot = legacyMediaRoot;
        this.mvcAsyncExecutor = mvcAsyncExecutor;
    }

    @Override
    public void configureAsyncSupport(AsyncSupportConfigurer configurer) {
        configurer.setTaskExecutor(mvcAsyncExecutor);
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/api/art-styles/**")
                .addResourceLocations("classpath:/static/art-styles/");

        String absolutePath = legacyMediaRoot.path().toString();
        // 确保路径以 / 结尾（Windows 的 \ 在 file: URI 中也需要处理）
        String resourceLocation = "file:" + absolutePath.replace('\\', '/') + "/";

        log.info("[WebMvcConfig] 静态资源映射: /media/** -> {}", resourceLocation);

        registry.addResourceHandler("/media/**")
                .addResourceLocations(resourceLocation);
    }
}
