package com.stonewu.fusion.service.ai.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AiModelRequestOptionsTests {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void appliesConfiguredEffortWithoutMutatingPersistedModel() throws Exception {
        AiModel persisted = AiModel.builder()
                .id(1L)
                .code("reasoning-model")
                .config("{\"temperature\":0.2}")
                .supportReasoning(true)
                .reasoningEffortLevels(List.of("max", "high", "low"))
                .build();

        AiModel effective = AiModelRequestOptions.withReasoningEffort(
                persisted, " high ", objectMapper);

        assertThat(effective).isNotSameAs(persisted);
        assertThat(objectMapper.readTree(effective.getConfig()).path("reasoningEffort").asText())
                .isEqualTo("high");
        assertThat(objectMapper.readTree(effective.getConfig()).path("temperature").asDouble())
                .isEqualTo(0.2);
        assertThat(objectMapper.readTree(persisted.getConfig()).has("reasoningEffort")).isFalse();
    }

    @Test
    void rejectsEffortThatIsNotConfiguredForModel() {
        AiModel model = AiModel.builder()
                .supportReasoning(true)
                .reasoningEffortLevels(List.of("high", "low"))
                .build();

        BusinessException exception = assertThrows(BusinessException.class,
                () -> AiModelRequestOptions.withReasoningEffort(model, "max", objectMapper));

        assertThat(exception.getCode()).isEqualTo(400);
    }
}
