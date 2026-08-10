package com.stonewu.fusion.config;

import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageConfigService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.AsyncTaskExecutor;
import org.springframework.web.servlet.config.annotation.AsyncSupportConfigurer;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.nio.file.Paths;

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

    @Value("${app.storage.local-base-path:./data/media}")
    private String localBasePath;

    private final StorageConfigService storageConfigService;
    private final AsyncTaskExecutor mvcAsyncExecutor;

    public WebMvcConfig(
            StorageConfigService storageConfigService,
            @Qualifier("mvcAsyncExecutor") AsyncTaskExecutor mvcAsyncExecutor) {
        this.storageConfigService = storageConfigService;
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

        // 优先从数据库中获取默认本地存储配置的 basePath
        String basePath = localBasePath;
        try {
            StorageConfig config = storageConfigService.getDefaultConfig();
            if (config != null && "local".equals(config.getType())
                    && config.getBasePath() != null && !config.getBasePath().isBlank()) {
                basePath = config.getBasePath();
            }
        } catch (Exception e) {
            log.debug("[WebMvcConfig] 数据库还未就绪，使用默认路径: {}", basePath);
        }

        // 转为绝对路径 file: URI
        String absolutePath = Paths.get(basePath).toAbsolutePath().normalize().toString();
        // 确保路径以 / 结尾（Windows 的 \ 在 file: URI 中也需要处理）
        String resourceLocation = "file:" + absolutePath.replace('\\', '/') + "/";

        log.info("[WebMvcConfig] 静态资源映射: /media/** -> {}", resourceLocation);

        registry.addResourceHandler("/media/**")
                .addResourceLocations(resourceLocation);
    }
}
