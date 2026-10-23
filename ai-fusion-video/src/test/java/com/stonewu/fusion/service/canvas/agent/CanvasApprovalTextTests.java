package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasApprovalTextTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasModelCatalog models = mock(CanvasModelCatalog.class);
    private final MediaResourceQueries media = mock(MediaResourceQueries.class);
    private final CanvasActor actor = new CanvasActor(1L, null);
    private final CanvasApprovalText display = new CanvasApprovalText(models, mock(ResourceQueryService.class), media, mock(CanvasMcpContracts.class), new CanvasCapabilityRegistry());

    @Test void fullAuthoredTextAndMeaningfulSettingsRemainVisibleWithoutInternalFields() {
        String content = "用户故事里写着：尚未实现这个愿望。".repeat(200);
        var value = json.valueToTree(Map.of("content", content, "configVersion", 8, "separator", "\n", "valueType", "text"));
        String result = display.describe(value, actor);
        assertThat(result).contains(content, "正文：", "分隔文字：", "内容类型：文本").doesNotContain("configVersion", "展开原始参数", "项配置");
        assertThat(display.describe(json.valueToTree(Map.of("content", "media://这只是故事中的文字")), actor)).isEqualTo("正文：media://这只是故事中的文字");
    }

    @Test void modelIdentityUsesTheSelectedModelNameAndPreservesGenerationSettings() {
        when(models.require(91)).thenReturn(new CanvasModelCatalog.Option(91, "短片图像模型", "image", "image", Map.of(), Map.of(), List.of(), null));
        String result = display.describe(json.valueToTree(Map.of("modelId", 91, "count", 2, "ratio", "16:9", "prompt", "温暖的灯塔")), actor);
        assertThat(result).contains("模型：短片图像模型", "候选数量：2", "画面比例：16:9", "温暖的灯塔").doesNotContain("modelId", "91");
    }

    @Test void dialogueIsReadableAndNoUnauthorizedMediaNameCanBeShown() throws Exception {
        var dialogue = json.readTree("[{\"type\":1,\"character_name\":\"阿青\",\"parenthetical\":\"轻声\",\"content\":\"是谁在等我？\",\"sortOrder\":0}]");
        assertThat(display.describe(dialogue, actor)).isEqualTo("阿青（轻声）：是谁在等我？");
        when(media.get("private-media", actor)).thenThrow(new BusinessException(403, "无权访问素材"));
        assertThatThrownBy(() -> display.describe(json.valueToTree(Map.of("resourceId", "private-media")), actor))
                .isInstanceOf(BusinessException.class).hasMessageContaining("无权访问素材");
    }

    @Test void projectStatusUsesItsOwnMeaningWithoutMutatingTheProposedFields() {
        var fields = json.valueToTree(Map.of("status", 1));
        assertThat(display.describeResourceFields(fields, CanvasValueType.PROJECT, actor)).contains("制作中").doesNotContain("正常");
        assertThat(fields.path("status").asInt()).isEqualTo(1);
    }

    @Test void everyDeclaredNodeSettingHasAReadableLabelAndUnknownProtocolCannotLeak() {
        var missing = new ArrayList<String>();
        for (var node : new CanvasCapabilityRegistry().list()) {
            for (String field : node.configFields().keySet()) {
                if (List.of("contractVersion", "resourceVersion").contains(field)) continue;
                try { assertThat(CanvasApprovalText.label(field)).as(node.kind() + "." + field).containsPattern("[\\u3400-\\u9fff]"); }
                catch (BusinessException absent) { missing.add(node.kind() + "." + field); }
                for (String choice : node.configFields().get(field).enumValues()) {
                    try { assertThat(display.describe(json.valueToTree(Map.of(field, choice)), actor)).doesNotContain(choice).containsPattern("[\\u3400-\\u9fff]"); }
                    catch (BusinessException absent) { missing.add(node.kind() + "." + field + "=" + choice); }
                }
            }
        }
        assertThat(missing).isEmpty();
        assertThatThrownBy(() -> CanvasApprovalText.label("unknownInternalProtocol"))
                .isInstanceOf(BusinessException.class).hasMessageNotContaining("unknownInternalProtocol");
    }
}
