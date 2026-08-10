package com.stonewu.fusion.service.system;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PresetArtStyleResourceResolverTests {

    private final PresetArtStyleResourceResolver resolver = new PresetArtStyleResourceResolver();

    @Test
    void toApiPathNormalizesPresetArtStyleUrls() {
        assertThat(resolver.toApiPath("/art-styles/anime_jp.jpg?version=1#preview"))
                .isEqualTo("/api/art-styles/anime_jp.jpg");
        assertThat(resolver.toApiPath("/api/art-styles/anime_cn.jpg"))
                .isEqualTo("/api/art-styles/anime_cn.jpg");
    }

    @Test
    void loadReadsPresetArtStyleFromClasspath() throws Exception {
        PresetArtStyleResourceResolver.PresetArtStyleResource resource = resolver.load("/art-styles/anime_jp.jpg");

        assertThat(resource.fileName()).isEqualTo("anime_jp.jpg");
        assertThat(resource.mimeType()).isEqualTo("image/jpeg");
        assertThat(resource.bytes()).isNotEmpty();
    }

    @Test
    void toApiPathRejectsTraversalSegments() {
        assertThat(resolver.toApiPath("/art-styles/../secret.png")).isNull();
    }
}