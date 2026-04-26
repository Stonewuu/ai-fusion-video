package com.stonewu.fusion.service.storage;

import org.junit.jupiter.api.Test;
import org.springframework.cache.annotation.Cacheable;

import static org.assertj.core.api.Assertions.assertThat;

class StorageConfigServiceTests {

    @Test
    void getDefaultConfigSkipsCachingNullResults() throws NoSuchMethodException {
        Cacheable cacheable = StorageConfigService.class
                .getMethod("getDefaultConfig")
                .getAnnotation(Cacheable.class);

        assertThat(cacheable).isNotNull();
        assertThat(cacheable.unless()).isEqualTo("#result == null");
    }
}