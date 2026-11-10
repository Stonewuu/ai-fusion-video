package com.stonewu.fusion.service.generation.image.strategy.gemini;

import com.fasterxml.jackson.databind.ObjectMapper;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.generation.ImageTask;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 参考图必须进入 Gemini parts；只保存最终图片，不能把思考缩略图当成结果。 */
class VertexGeminiImageProtocolTests {
    @Test
    void preservesExplicitNativeRatioAndTierForPanoramasAndSmallImages() throws Exception {
        var task = ImageTask.builder().aspectRatio("1:8").resolution("1K").width(384).height(3072).build();
        var output = VertexGeminiImageProtocol.outputOptions(task, new int[]{384, 3072}, JSONUtil.createObj());
        var request = new ObjectMapper().readTree(VertexGeminiImageProtocol.requestBody("全景", output[0], output[1], List.of()));
        assertThat(request.at("/generationConfig/imageConfig/imageSize").asText()).isEqualTo("1K");
        assertThat(request.at("/generationConfig/imageConfig/aspectRatio").asText()).isEqualTo("1:8");
        task.setResolution("512");
        assertThat(VertexGeminiImageProtocol.outputOptions(task, new int[]{192, 1536}, JSONUtil.createObj())[1]).isEqualTo("512");
        var config = JSONUtil.createObj().set("nativeImageSize", false);
        assertThat(VertexGeminiImageProtocol.outputOptions(task, new int[]{192, 1536}, config)[1]).isNull();
    }
    @Test
    void sendsEveryReferenceAndNativeImageConfiguration() throws Exception {
        var root = new ObjectMapper().readTree(VertexGeminiImageProtocol.requestBody("保持人物一致", 2048, 1152,
                List.of("data:image/png;base64,YQ==", "data:image/jpeg;base64,Yg==")));
        var parts = root.path("contents").get(0).path("parts");
        assertThat(parts.size()).isEqualTo(3);
        assertThat(parts.get(0).path("inlineData").path("data").asText()).isEqualTo("YQ==");
        assertThat(parts.get(1).path("inlineData").path("mimeType").asText()).isEqualTo("image/jpeg");
        assertThat(root.at("/generationConfig/imageConfig/aspectRatio").asText()).isEqualTo("16:9");
        assertThat(root.at("/generationConfig/imageConfig/imageSize").asText()).isEqualTo("2K");
        assertThatThrownBy(() -> VertexGeminiImageProtocol.requestBody("test", 100, 900, List.of()))
                .hasMessageContaining("比例");
    }

    @Test
    void excludesThoughtImagesAndRejectsTextOnlyResponses() {
        assertThat(VertexGeminiImageProtocol.imageOutputs("""
                {"candidates":[{"content":{"parts":[
                  {"thought":true,"inlineData":{"mimeType":"image/png","data":"dGhvdWdodA=="}},
                  {"text":"图片如下"},
                  {"inlineData":{"mimeType":"image/png","data":"ZmluYWw="}}
                ]}}]}
                """)).containsExactly("data:image/png;base64,ZmluYWw=");
        assertThatThrownBy(() -> VertexGeminiImageProtocol.imageOutputs("{\"candidates\":[]}"))
                .hasMessageContaining("未返回图片");
    }
}
