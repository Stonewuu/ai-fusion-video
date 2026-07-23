package com.stonewu.fusion.service.generation.image.strategy.openai;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolContext;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageRequest;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import okhttp3.OkHttpClient;
import okio.Buffer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

class OpenAiImageProtocolTests {

    private final OpenAiCompatibleImageProtocolSupport support =
            new OpenAiCompatibleImageProtocolSupport(
                    mock(ModelPresetService.class),
                    mock(MediaStorageService.class),
                    mock(StorageConfigService.class),
                    new PresetArtStyleResourceResolver(),
                    new OkHttpClient()
            );
    private final OpenAiImageProtocolAdapter adapter = new OpenAiImageProtocolAdapter(support);

    @Test
    void generationsRequestContainsOnlyOfficialOpenAiFields() throws IOException {
        JSONObject config = JSONUtil.parseObj("""
                {
                  "quality": "high",
                  "background": "opaque",
                  "moderation": "auto",
                  "outputFormat": "png",
                  "outputCompression": 90,
                  "responseFormat": "b64_json",
                  "style": "natural",
                  "user": "user-123",
                  "resolution": "2K",
                  "maskUrl": "https://example.com/mask.png"
                }
                """);

        JSONObject body = JSONUtil.parseObj(readBody(adapter.buildRequest(
                context("gpt-image-2", List.of(), config))));

        assertThat(adapter.getProtocol()).isEqualTo("openai");
        assertThat(body.keySet()).contains(
                "model", "prompt", "n", "size", "quality", "background", "moderation",
                "output_format", "output_compression", "response_format", "style", "user");
        assertThat(body.keySet()).doesNotContain("image_urls", "resolution", "mask_url", "extra_body");
    }

    @Test
    void editsRequestOmitsCompatibilityOnlyFormFields() throws IOException {
        JSONObject config = JSONUtil.parseObj("""
                {
                  "quality": "high",
                  "inputFidelity": "high",
                  "resolution": "2K",
                  "style": "natural"
                }
                """);

        String body = readBody(adapter.buildRequest(context(
                "gpt-image-1.5",
                List.of("data:image/png;base64,cmVmZXJlbmNl"),
                config
        )));

        assertThat(body).contains("name=\"image[]\"");
        assertThat(body).contains("name=\"input_fidelity\"");
        assertThat(body).doesNotContain("name=\"resolution\"");
        assertThat(body).doesNotContain("name=\"style\"");
    }

    @Test
    void gptImage2EditsOmitUnsupportedInputFidelity() throws IOException {
        JSONObject config = JSONUtil.parseObj("""
                {
                  "inputFidelity": "high"
                }
                """);

        String body = readBody(adapter.buildRequest(context(
                "gpt-image-2",
                List.of("data:image/png;base64,cmVmZXJlbmNl"),
                config
        )));

        assertThat(body).contains("name=\"image[]\"");
        assertThat(body).doesNotContain("name=\"input_fidelity\"");
    }

    @Test
    void responseParserDoesNotAcceptCompatibilityOnlyImageUrlAlias() {
        assertThatThrownBy(() -> adapter.parseImageUrls(null, """
                {"data":[{"image_url":"https://example.com/result.png"}]}
                """))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("未找到 url 或 b64_json");
    }

    private OpenAiCompatibleImageProtocolContext context(String modelCode,
                                                          List<String> imageUrls,
                                                          JSONObject config) {
        return new OpenAiCompatibleImageProtocolContext(
                null,
                ApiConfig.builder()
                        .platform("openai_compatible")
                        .apiUrl("https://api.openai.com")
                        .apiKey("test-key")
                        .build(),
                modelCode,
                "a test prompt",
                1024,
                1024,
                1,
                imageUrls,
                config,
                false
        );
    }

    private String readBody(OpenAiCompatibleImageRequest request) throws IOException {
        Buffer buffer = new Buffer();
        request.body().writeTo(buffer);
        return new String(buffer.readByteArray(), StandardCharsets.UTF_8);
    }
}
