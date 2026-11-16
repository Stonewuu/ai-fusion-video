package com.stonewu.fusion.service.script;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

class ScriptContextProjectionTests {
    private final ObjectMapper json = new ObjectMapper();

    @Test
    void canvasResourceContextOmitsSourceButKeepsSynopsisAndVersion() {
        var original = json.createObjectNode().put("id", 12).put("version", 4).put("rawContent", "小说全文")
                .put("content", "剧本正文").put("synopsis", "本集剧情");
        var projected = ScriptContextProjection.summary(CanvasValueType.SCRIPT_EPISODE, original);
        assertThat(projected.toString()).doesNotContain("小说全文", "剧本正文");
        assertThat(projected.path("synopsis").asText()).isEqualTo("本集剧情");
        assertThat(projected.path("version").asInt()).isEqualTo(4);
        assertThat(original.path("rawContent").asText()).isEqualTo("小说全文");
    }

    @Test
    void nestedCanvasInputsKeepSceneDialogueAndUserInstructions() {
        var input = json.createObjectNode().put("instruction", "生成当前集分镜");
        var inputs = input.putArray("inputs");
        inputs.addObject().put("rawContent", "第一集原文").put("synopsis", "概述");
        inputs.addObject().put("projectId", 1).put("totalEpisodes", 1).put("content", "旧版总剧本正文");
        inputs.addObject().put("type", 1).put("content", "场次里的对白");
        ScriptContextProjection.omitSource(input);
        assertThat(input.toString()).doesNotContain("rawContent", "第一集原文", "旧版总剧本正文");
        assertThat(input.toString()).contains("概述", "场次里的对白", "生成当前集分镜");
    }
}
