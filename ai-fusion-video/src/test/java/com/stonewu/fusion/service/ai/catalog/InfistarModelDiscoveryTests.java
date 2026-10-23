package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 保护真实 HTTP 发现边界：密钥权限、接口分类与平台能力必须同时满足。 */
class InfistarModelDiscoveryTests {
    private final ModelPresetService presets = new ModelPresetService();
    private final CopyOnWriteArrayList<String> headers = new CopyOnWriteArrayList<>();
    private HttpServer server;
    private ApiConfig config;
    private ProviderDefinition provider;
    private String account = "";
    private String metadata = "";
    private int accountStatus = 200;

    @BeforeEach
    void start() throws Exception {
        presets.init();
        provider = new ProviderCatalogService(new ObjectMapper()).require("infistar");
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            headers.add(exchange.getRequestHeaders().getFirst("Authorization"));
            boolean models = exchange.getRequestURI().getPath().equals("/v1/models");
            byte[] bytes = (models ? account : metadata).getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(models ? accountStatus : 200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        config = ApiConfig.builder().apiKey("test-discovery-key")
                .apiUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/v1").build();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    @Test
    void intersectsAccountAndCatalogWithoutGuessingFromNamesOrLeakingPricing() {
        account = """
                {"data":[{"id":"MiniMax-H3"},{"id":"gpt-image-2"},{"id":"future-video"},
                {"id":"looks-like-image"},{"id":"response-only"},{"id":"native-claude"},
                {"id":"native-gemini"},{"id":"not-in-square"},{"id":"MiniMax-H3"},
                {"id":"account-denied","supported_endpoint_types":[]}]}
                """;
        metadata = """
                {"success":true,"data":[
                {"model_name":"MiniMax-H3","supported_endpoint_types":["video"],"group_prices":{"secret":10}},
                {"model_name":"gpt-image-2","supported_endpoint_types":["image"]},
                {"model_name":"future-video","supported_endpoint_types":["video"]},
                {"model_name":"looks-like-image","supported_endpoint_types":["chat"]},
                {"model_name":"response-only","supported_endpoint_types":["responses"]},
                {"model_name":"native-claude","supported_endpoint_types":["claude"]},
                {"model_name":"native-gemini","supported_endpoint_types":["gemini"]},
                {"model_name":"not-in-account","supported_endpoint_types":["chat"]},
                {"model_name":"account-denied","supported_endpoint_types":["chat"]}],
                "supported_endpoint":{
                "video":{"method":"POST","path":"/v1/videos"},
                "image":{"method":"POST","path":"/v1/images/generations"},
                "chat":{"method":"POST","path":"/v1/chat/completions"},
                "responses":{"method":"POST","path":"/v1/responses"},
                "claude":{"method":"POST","path":"/v1/messages"},
                "gemini":{"method":"POST","path":"/v1beta/models/{model}:generateContent"}}}
                """;
        var result = new InfistarModelDiscovery(presets).discover(config, provider);
        assertThat(result.models()).extracting(ProviderModelCandidate::id).containsExactly(
                "MiniMax-H3", "gpt-image-2", "future-video", "looks-like-image", "response-only",
                "native-claude", "native-gemini");
        assertThat(result.models().get(0).capabilityPresetCode()).isEqualTo("infistar:MiniMax-H3");
        assertThat(result.models().get(1).capabilityPresetCode()).isEqualTo("infistar:gpt-image-2");
        assertThat(result.models().get(2).capabilityPresetCode()).isEqualTo("infistar:common-video");
        assertThat(result.models().get(2).configurationHint()).isNull();
        assertThat(result.models().get(2).name()).isEqualTo("future-video");
        assertThat(result.models().get(3).modelType()).isEqualTo(1);
        assertThat(JSONUtil.parseObj(result.models().get(4).config()).getStr("apiMode")).isEqualTo("responses");
        assertThat(result.models().get(5).modelProtocol()).isEqualTo("anthropic");
        assertThat(result.models().get(6).modelProtocol()).isEqualTo("gemini");
        assertThat(JSONUtil.toJsonStr(result)).doesNotContain("group_prices", "secret", "test-discovery-key");
        assertThat(headers).containsOnly("Bearer test-discovery-key");
        var scoped = JSONUtil.parseObj(presets.getPresetConfig("infistar:MiniMax-H3"));
        assertThat(scoped.getBool("supportLastFrame")).isFalse();
        assertThat(presets.findPresetCode("MiniMax-H3", 3)).isNotEqualTo("infistar:MiniMax-H3");
    }

    @Test
    void imageOutputUsingConversationEndpointsIsNotAdvertisedAsText() {
        account = """
                {"data":[{"id":"gemini-nano-banana-2.1"},{"id":"gemini-3.1-flash-image"},
                {"id":"visual-chat"},{"id":"gpt-image-2"}]}
                """;
        metadata = """
                {"success":true,"data":[
                {"model_name":"gemini-nano-banana-2.1","tags":"图像理解,图片生成","supported_endpoint_types":["gemini"]},
                {"model_name":"gemini-3.1-flash-image","tags":"图片生成","supported_endpoint_types":["chat"]},
                {"model_name":"visual-chat","tags":"图像理解,文本生成","supported_endpoint_types":["chat"]},
                {"model_name":"gpt-image-2","tags":"图片生成","supported_endpoint_types":["image"]}],
                "supported_endpoint":{
                "image":{"method":"POST","path":"/v1/images/generations"},
                "chat":{"method":"POST","path":"/v1/chat/completions"},
                "gemini":{"method":"POST","path":"/v1beta/models/{model}:generateContent"}}}
                """;
        var result = new InfistarModelDiscovery(presets).discover(config, provider);
        assertThat(result.models()).extracting(ProviderModelCandidate::id)
                .containsExactly("visual-chat", "gpt-image-2");
        assertThat(result.models()).extracting(ProviderModelCandidate::modelType).containsExactly(1, 2);
    }

    @Test
    void authenticationAndMalformedCatalogDoNotBecomeAnEmptySuccess() {
        accountStatus = 401;
        account = "{\"error\":{\"message\":\"invalid key\"}}";
        assertThatThrownBy(() -> new InfistarModelDiscovery(presets).discover(config, provider))
                .hasMessageContaining("401");
        accountStatus = 200;
        account = "{\"data\":[]}";
        metadata = "{\"success\":true,\"data\":[]}";
        assertThatThrownBy(() -> new InfistarModelDiscovery(presets).discover(config, provider))
                .hasMessageContaining("缺少接口说明");
        metadata = "{\"success\":true,\"data\":[],\"supported_endpoint\":{}}";
        assertThat(new InfistarModelDiscovery(presets).discover(config, provider).models()).isEmpty();
    }
}
