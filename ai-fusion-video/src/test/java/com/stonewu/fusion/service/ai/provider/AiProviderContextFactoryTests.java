package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class AiProviderContextFactoryTests {

    @Test
    void createForApiConfigNormalizesLegacyOpenAiPlatform() {
        AiProviderContextFactory factory = new AiProviderContextFactory(null);

        AiProviderContext context = factory.createForApiConfig(
                ApiConfig.builder().platform("openai").apiUrl("https://api.openai.com").build());

        assertThat(context.getPlatform()).isEqualTo("openai_compatible");
    }

    @Test
    void createForApiConfigDetectsOpenAiUrlAsOpenAiCompatible() {
        AiProviderContextFactory factory = new AiProviderContextFactory(null);

        AiProviderContext context = factory.createForApiConfig(
                ApiConfig.builder().apiUrl("https://api.openai.com").build());

        assertThat(context.getPlatform()).isEqualTo("openai_compatible");
    }
}