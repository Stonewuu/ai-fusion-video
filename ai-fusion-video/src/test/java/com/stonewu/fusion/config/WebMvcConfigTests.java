package com.stonewu.fusion.config;

import com.stonewu.fusion.service.storage.StorageConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.core.task.AsyncTaskExecutor;
import org.springframework.web.servlet.config.annotation.AsyncSupportConfigurer;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class WebMvcConfigTests {

    @Test
    void usesTheBoundedMvcExecutorForReactiveAndSseRequests() {
        AsyncTaskExecutor executor = mock(AsyncTaskExecutor.class);
        WebMvcConfig config = new WebMvcConfig(
                mock(StorageConfigService.class), executor);
        AsyncSupportConfigurer asyncSupport = mock(AsyncSupportConfigurer.class);

        config.configureAsyncSupport(asyncSupport);

        verify(asyncSupport).setTaskExecutor(executor);
    }
}
