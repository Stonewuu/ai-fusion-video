package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiMultimodalInputVO;
import com.stonewu.fusion.entity.ai.AiModel;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AiModelMultimodalCapabilitiesTests {

    @Test
    void normalizesCanonicalTypesAndDerivesVisionCompatibilityFlag() {
        AiModel model = AiModel.builder()
                .modelType(1)
                .multimodalInputTypes(List.of(" Image ", "video"))
                .multimodalInputTransports(Map.of(
                        "image", List.of("BASE64", "url"),
                        "video", List.of("url")))
                .build();

        AiModelMultimodalCapabilities.normalizeModel(model);

        assertThat(model.getMultimodalInputTypes()).containsExactly("image", "video");
        assertThat(model.getMultimodalInputTransports().get("image"))
                .containsExactly("base64", "url");
        assertThat(model.getSupportVision()).isTrue();
    }

    @Test
    void rejectsTransportKeysThatDoNotMatchEnabledTypes() {
        AiModel model = AiModel.builder()
                .modelType(1)
                .multimodalInputTypes(List.of("image"))
                .multimodalInputTransports(Map.of("video", List.of("url")))
                .build();

        assertThatThrownBy(() -> AiModelMultimodalCapabilities.normalizeModel(model))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未启用");
    }

    @Test
    void validatesBase64InputAgainstSelectedModelCapability() {
        AiModel model = AiModel.builder()
                .modelType(1)
                .status(1)
                .multimodalInputTypes(List.of("image"))
                .multimodalInputTransports(Map.of("image", List.of("base64")))
                .build();
        AiMultimodalInputVO input = new AiMultimodalInputVO();
        input.setId("image-1");
        input.setName("sample.png");
        input.setInputType("image");
        input.setMimeType("image/png");
        input.setTransport("base64");
        input.setData("aGVsbG8=");
        input.setSize(5L);

        AiModelMultimodalCapabilities.validateInputs(model, List.of(input));

        assertThat(input.getSize()).isEqualTo(5L);
        assertThat(input.getMimeType()).isEqualTo("image/png");
    }
}
