package com.stonewu.fusion.service.generation.image.strategy.newapi;

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
import static org.mockito.Mockito.mock;

class NewApiImageProtocolTests {

    private final OpenAiCompatibleImageProtocolSupport support =
            new OpenAiCompatibleImageProtocolSupport(
                    mock(ModelPresetService.class),
                    mock(MediaStorageService.class),
                    mock(StorageConfigService.class),
                    new PresetArtStyleResourceResolver(),
                    new OkHttpClient()
            );
    private final NewApiImageProtocolAdapter adapter = new NewApiImageProtocolAdapter(support);

    @Test
    void urlCapableEditsUseJsonImageReferences() throws IOException {
        JSONObject config = JSONUtil.parseObj("""
                {
                  "referenceImageInputFormats": ["url", "data_uri"],
                  "resolution": "2K",
                  "style": "natural"
                }
                """);

        OpenAiCompatibleImageRequest request = adapter.buildRequest(context(
                List.of("https://cdn.example.com/reference.png"),
                config
        ));
        JSONObject body = JSONUtil.parseObj(readBody(request));

        assertThat(request.body().contentType().toString()).startsWith("application/json");
        assertThat(body.getJSONArray("images").getJSONObject(0).getStr("image_url"))
                .isEqualTo("https://cdn.example.com/reference.png");
        assertThat(body.getStr("resolution")).isEqualTo("2K");
        assertThat(body.getStr("style")).isEqualTo("natural");
    }

    @Test
    void dataUriOnlyEditsKeepMultipartUpload() throws IOException {
        JSONObject config = JSONUtil.parseObj("""
                {
                  "referenceImageInputFormats": ["data_uri"]
                }
                """);

        OpenAiCompatibleImageRequest request = adapter.buildRequest(context(
                List.of("data:image/png;base64,cmVmZXJlbmNl"),
                config
        ));
        String body = readBody(request);

        assertThat(request.body().contentType().toString()).startsWith("multipart/form-data");
        assertThat(body).contains("name=\"image[]\"");
    }

    @Test
    void generationsReferenceModeKeepsImageUrlsContract() throws IOException {
        JSONObject config = JSONUtil.parseObj("""
                {
                  "referenceImageInputFormats": ["url"],
                  "referenceImagesViaGenerations": true
                }
                """);

        JSONObject body = JSONUtil.parseObj(readBody(adapter.buildRequest(context(
                List.of("https://cdn.example.com/reference.png"),
                config
        ))));

        assertThat(body.getJSONArray("image_urls").getStr(0))
                .isEqualTo("https://cdn.example.com/reference.png");
        assertThat(body.keySet()).doesNotContain("images");
    }

    private OpenAiCompatibleImageProtocolContext context(List<String> imageUrls, JSONObject config) {
        return new OpenAiCompatibleImageProtocolContext(
                null,
                ApiConfig.builder()
                        .platform("openai_compatible")
                        .apiUrl("https://gateway.example.com")
                        .apiKey("test-key")
                        .build(),
                "gpt-image-2",
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
