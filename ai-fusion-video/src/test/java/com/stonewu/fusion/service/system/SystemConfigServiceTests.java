package com.stonewu.fusion.service.system;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.system.SystemConfig;
import com.stonewu.fusion.mapper.system.SystemConfigMapper;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SystemConfigServiceTests {

    private final SystemConfigMapper systemConfigMapper = mock(SystemConfigMapper.class);
    private final PresetArtStyleResourceResolver presetResolver = mock(PresetArtStyleResourceResolver.class);
    private final SystemConfigService service = new SystemConfigService(systemConfigMapper, presetResolver);

    @Test
    void resourceBaseUrlDoesNotReadSiteBaseUrlWhenItIsMissing() {
        when(systemConfigMapper.selectOne(any())).thenReturn(null);

        assertThat(service.getPublicResourceBaseUrl()).isNull();

        verify(systemConfigMapper).selectOne(any());
    }

    @Test
    void resolvesRelativeMediaAgainstExplicitResourceBaseUrl() {
        when(presetResolver.isPresetArtStylePath("/media/reference.png")).thenReturn(false);
        when(systemConfigMapper.selectOne(any())).thenReturn(
                SystemConfig.builder().configValue("https://api.example.com").build());

        assertThat(service.resolvePublicUrl("/media/reference.png"))
                .isEqualTo("https://api.example.com/media/reference.png");
    }

    @Test
    void doesNotResolvePresetResourceWithoutResourceBaseUrl() {
        when(presetResolver.isPresetArtStylePath("/art-styles/demo.png")).thenReturn(true);
        when(presetResolver.toApiPath("/art-styles/demo.png")).thenReturn("/api/art-styles/demo.png");
        when(systemConfigMapper.selectOne(any())).thenReturn(null);

        assertThat(service.resolvePublicUrl("/art-styles/demo.png")).isNull();
    }

    @Test
    void rejectsResourceBaseUrlEndingWithApiPath() {
        assertThatThrownBy(() -> service.setValue(
                SystemConfigService.RESOURCE_BASE_URL_KEY,
                "https://api.example.com/api"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("不能包含末尾 /api");

        verify(systemConfigMapper, never()).selectOne(any());
    }

    @Test
    void normalizesConfiguredPublicUrlBeforeSaving() {
        when(systemConfigMapper.selectOne(any())).thenReturn(null);
        ArgumentCaptor<SystemConfig> captor = ArgumentCaptor.forClass(SystemConfig.class);

        service.setValue(SystemConfigService.SITE_BASE_URL_KEY, " https://app.example.com/// ");

        verify(systemConfigMapper).insert(captor.capture());
        assertThat(captor.getValue().getConfigValue()).isEqualTo("https://app.example.com");
    }
}
