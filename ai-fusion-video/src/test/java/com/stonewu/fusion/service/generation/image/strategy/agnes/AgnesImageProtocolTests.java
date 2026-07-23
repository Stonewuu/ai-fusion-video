package com.stonewu.fusion.service.generation.image.strategy.agnes;

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
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class AgnesImageProtocolTests {

    private final OpenAiCompatibleImageProtocolSupport support =
            new OpenAiCompatibleImageProtocolSupport(
                    mock(ModelPresetService.class),
                    mock(MediaStorageService.class),
                    mock(StorageConfigService.class),
                    new PresetArtStyleResourceResolver(),
                    new OkHttpClient()
            );
    private final AgnesImageProtocolAdapter agnesAdapter =
            new AgnesImageProtocolAdapter(support);
    @Test
    void buildsSharedAgnesSchemaWithVersionSpecificSizeConfiguration() throws IOException {
        JSONObject body20 = JSONUtil.parseObj(readBody(agnesAdapter.buildRequest(
                context("agnes-image-2.0-flash", 1024, 768, agnes20Config()))));
        JSONObject body21 = JSONUtil.parseObj(readBody(agnesAdapter.buildRequest(
                context("agnes-image-2.1-flash", 2624, 1472, agnes21Config()))));

        assertThat(body20.getStr("size")).isEqualTo("1024x768");
        assertThat(body20.containsKey("ratio")).isFalse();
        assertThat(body21.getStr("size")).isEqualTo("2K");
        assertThat(body21.getStr("ratio")).isEqualTo("16:9");

        for (JSONObject body : List.of(body20, body21)) {
            assertThat(body.containsKey("n")).isFalse();
            assertThat(body.containsKey("response_format")).isFalse();
            assertThat(body.containsKey("tags")).isFalse();
            assertThat(body.getJSONObject("extra_body").getStr("response_format")).isEqualTo("url");
            assertThat(body.getJSONObject("extra_body").getJSONArray("image").toList(String.class))
                    .containsExactly("data:image/png;base64,cmVmZXJlbmNl");
        }
    }

    private OpenAiCompatibleImageProtocolContext context(String modelCode,
                                                          int width,
                                                          int height,
                                                          JSONObject config) {
        return new OpenAiCompatibleImageProtocolContext(
                null,
                ApiConfig.builder()
                        .platform("openai_compatible")
                        .apiUrl("https://apihub.agnes-ai.com")
                        .apiKey("test-key")
                        .build(),
                modelCode,
                "preserve the composition",
                width,
                height,
                1,
                List.of("data:image/png;base64,cmVmZXJlbmNl"),
                config,
                false
        );
    }

    private JSONObject agnes20Config() {
        return JSONUtil.parseObj("""
                {
                  "imageProtocol": "agnes",
                  "responseFormat": "url",
                  "supportedResolutions": ["1024x768"],
                  "supportedSizes": {"standard": {"4:3": "1024x768"}}
                }
                """);
    }

    private JSONObject agnes21Config() {
        return JSONUtil.parseObj("""
                {
                  "imageProtocol": "agnes",
                  "responseFormat": "url",
                  "supportedResolutions": ["1K", "2K"],
                  "supportedSizes": {"2K": {"16:9": "2624x1472"}}
                }
                """);
    }

    private String readBody(OpenAiCompatibleImageRequest request) throws IOException {
        Buffer buffer = new Buffer();
        request.body().writeTo(buffer);
        return buffer.readUtf8();
    }
}
