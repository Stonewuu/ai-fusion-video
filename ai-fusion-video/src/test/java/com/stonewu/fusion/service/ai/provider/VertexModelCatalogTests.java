package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 保护官方目录分页、模型类型和鉴权错误，不访问外部 Google 服务。 */
class VertexModelCatalogTests {
    @Test
    void followsPageTokensAndReturnsOnlyGenerationModels() {
        var requests = new ArrayList<String>();
        var result = new VertexModelCatalog().load(VertexModelCatalog.ENDPOINT, url -> {
            requests.add(url);
            return requests.size() == 1 ? """
                    {"publisherModels":[
                      {"name":"publishers/google/models/gemini-3.8-flash"},
                      {"name":"publishers/google/models/gemini-omni-1.1-flash-preview"},
                      {"name":"publishers/google/models/gemini-3.8-flash-tts"},
                      {"name":"publishers/google/models/gemini-embedding-2"},
                      {"name":"publishers/google/models/veo-3"}
                    ],"nextPageToken":"a+/= b"}
                    """ : """
                    {"publisherModels":[
                      {"name":"publishers/google/models/gemini-3.8-flash"},
                      {"name":"publishers/google/models/gemini-omni-1.2-flash"},
                      {"name":"publishers/google/models/gemini-nano-banana-2.1"},
                      {"name":"publishers/google/models/imagen-4.0-generate-001"}
                    ]}
                    """;
        });
        assertThat(requests).hasSize(2);
        assertThat(requests.get(1)).contains("pageToken=a%2B%2F%3D%20b");
        assertThat(result).extracting(RemoteModelVO::getId).containsExactly(
                "gemini-3.8-flash", "gemini-omni-1.1-flash-preview", "gemini-3.8-flash-tts",
                "veo-3", "gemini-omni-1.2-flash", "gemini-nano-banana-2.1",
                "imagen-4.0-generate-001");
        assertThat(result).extracting(RemoteModelVO::getModelType).containsExactly(1, 3, 4, 3, 3, 2, 2);
        assertThat(result).allSatisfy(model -> assertThat(model.getModelProtocol()).isEqualTo("vertex_ai"));
    }

    @Test
    void rejectsRepeatedTokensAndFailedLaterPagesInsteadOfReturningPartialSuccess() {
        var catalog = new VertexModelCatalog();
        assertThatThrownBy(() -> catalog.load(VertexModelCatalog.ENDPOINT,
                url -> "{\"publisherModels\":[],\"nextPageToken\":\"same\"}"))
                .hasMessageContaining("重复分页");
        assertThatThrownBy(() -> catalog.load(VertexModelCatalog.ENDPOINT, url -> {
            if (url.contains("pageToken")) {
                throw new BusinessException("HTTP 403");
            }
            return "{\"publisherModels\":[],\"nextPageToken\":\"next\"}";
        })).hasMessageContaining("403");
    }

    @Test
    void apiKeyCatalogFailureExplainsOfficialLimitationWithoutSwitchingToStoredJson() {
        var config = ApiConfig.builder().platform("vertex_ai_api_key").apiKey("fixture-key")
                .appSecret("invalid-stored-json").build();
        var provider = new VertexAiProvider() {
            @Override
            protected String executeGet(String url, Map<String, String> headers, ApiConfig api) {
                assertThat(headers).containsExactlyEntriesOf(Map.of("x-goog-api-key", "fixture-key"));
                throw new BusinessException("HTTP 401: API keys are not supported by this API.");
            }
        };
        assertThatThrownBy(() -> provider.listRemoteModels(AiProviderContext.builder().apiConfig(config).build()))
                .hasMessageContaining("目录接口不接受 API Key");
    }

    @Test
    void explicitAuthenticationControlsHeaderAndEndpointWithoutCredentialFallback() {
        var config = ApiConfig.builder().platform("vertex_ai_api_key").apiKey("fixture-key")
                .appId("old-project").apiUrl("us-central1").appSecret("invalid-json").build();
        assertThat(VertexAiAuthentication.headers(config)).containsOnlyKeys("x-goog-api-key");
        assertThat(VertexAiAuthentication.modelUrl(config, "gemini-test", "generateContent"))
                .isEqualTo("https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-test:generateContent");
        config.setApiKey("");
        assertThatThrownBy(() -> VertexAiAuthentication.headers(config)).hasMessageContaining("API Key");
        config.setPlatform("vertex_ai");
        config.setApiKey("unused-key");
        config.setApiUrl("global");
        assertThat(VertexAiAuthentication.modelUrl(config, "gemini-test", "generateContent"))
                .isEqualTo("https://aiplatform.googleapis.com/v1/projects/old-project/locations/global/publishers/google/models/gemini-test:generateContent");
        assertThatThrownBy(() -> VertexAiAuthentication.headers(config)).hasMessageContaining("凭据无法加载");
    }
}
