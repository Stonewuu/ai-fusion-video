package com.stonewu.fusion.build;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.volcengine.ark.runtime.model.images.generation.GenerateImagesRequest;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ArkJacksonCompatibilityTests {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void arkImageRequestSerializesBuilderValuesWithJackson() throws Exception {
        GenerateImagesRequest request = GenerateImagesRequest.builder()
                .model("seedream-test")
                .prompt("test")
                .size("1024x1024")
                .watermark(false)
                .build();

        JsonNode json = objectMapper.readTree(objectMapper.writeValueAsString(request));

        assertThat(json.path("model").asText()).isEqualTo("seedream-test");
        assertThat(json.path("prompt").asText()).isEqualTo("test");
        assertThat(json.path("size").asText()).isEqualTo("1024x1024");
        JsonNode watermark = json.get("watermark");
        assertThat(watermark).isNotNull();
        assertThat(watermark.isBoolean()).isTrue();
        assertThat(watermark.booleanValue()).isFalse();
    }
}
