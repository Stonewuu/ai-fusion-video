package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.controller.ai.vo.AiMultimodalInputVO;
import com.stonewu.fusion.entity.ai.AgentMessage;
import io.agentscope.core.message.AudioBlock;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.message.VideoBlock;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentScopeMessageMapperTests {

    @Test
    void mapsTextToStrongUserMessageWithoutRoleMutation() {
        AgentScopeMessageMapper mapper = new AgentScopeMessageMapper();

        UserMessage message = mapper.toUserMessage(" hello ");

        assertThat(message.getContent()).singleElement()
                .isInstanceOfSatisfying(TextBlock.class,
                        block -> assertThat(block.getText()).isEqualTo(" hello "));
        assertThat(mapper.toUserMessages("hello")).singleElement()
                .isInstanceOfSatisfying(UserMessage.class, mapped ->
                        assertThat(mapped.getContent()).singleElement()
                                .isInstanceOfSatisfying(TextBlock.class,
                                        block -> assertThat(block.getText()).isEqualTo("hello")));
        assertThatThrownBy(() -> mapper.toUserMessage("  "))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void mapsEachInputTypeToItsOfficialAgentScopeBlock() {
        AgentScopeMessageMapper mapper = new AgentScopeMessageMapper();
        AiMultimodalInputVO image = input(
                "image-1", "sample.png", "image", "image/png", "base64", null, "aGVsbG8=");
        AiMultimodalInputVO video = input(
                "video-1", "sample.mp4", "video", "video/mp4", "url", "https://example.com/sample.mp4", null);
        AiMultimodalInputVO audio = input(
                "audio-1", "sample.wav", "audio", "audio/wav", "base64", null, "aGVsbG8=");
        AiMultimodalInputVO file = input(
                "file-1", "sample.pdf", "file", "application/pdf", "url", "https://example.com/sample.pdf", null);

        UserMessage message = mapper.toUserMessage("describe", List.of(image, video, audio, file));

        assertThat(message.getContent()).hasSize(5);
        assertThat(message.getContent().get(1)).isInstanceOfSatisfying(ImageBlock.class, block ->
            assertThat(block.getSource()).isInstanceOfSatisfying(Base64Source.class, source -> {
                assertThat(source.getMediaType()).isEqualTo("image/png");
                assertThat(source.getData()).isEqualTo("aGVsbG8=");
            }));
        assertThat(message.getContent().get(2)).isInstanceOfSatisfying(VideoBlock.class, block ->
                assertThat(block.getSource()).isInstanceOfSatisfying(URLSource.class, source -> {
                    assertThat(source.getUrl()).isEqualTo("https://example.com/sample.mp4");
                    assertThat(source.getMimeType()).isEqualTo("video/mp4");
                }));
        assertThat(message.getContent().get(3)).isInstanceOfSatisfying(AudioBlock.class, block ->
                assertThat(block.getSource()).isInstanceOf(Base64Source.class));
        assertThat(message.getContent().get(4)).isInstanceOfSatisfying(DataBlock.class, block -> {
            assertThat(block.getId()).isEqualTo("file-1");
            assertThat(block.getName()).isEqualTo("sample.pdf");
            assertThat(block.getSource()).isInstanceOf(URLSource.class);
        });
    }

    @Test
    void rebuildsDurableContinuationContextWithoutLeavingPendingToolCalls() {
        AgentScopeMessageMapper mapper = new AgentScopeMessageMapper();
        List<Msg> recovered = mapper.toRecoveredContinuationMessages(List.of(
                AgentMessage.builder().role("user").content("生成第一集分镜").build(),
                AgentMessage.builder()
                        .role("tool")
                        .toolCallId("call-1")
                        .toolName("get_script_episode")
                        .toolStatus("running")
                        .content("{\"scriptEpisodeId\":13}")
                        .build(),
                AgentMessage.builder()
                        .role("tool")
                        .toolCallId("call-1")
                        .toolName("get_script_episode")
                        .toolStatus("success")
                        .content("{\"title\":\"第一集\"}")
                        .build(),
                AgentMessage.builder()
                        .role("tool")
                        .toolCallId("call-2")
                        .toolName("save_storyboard_scene_shots")
                        .toolStatus("running")
                        .content("{\"sceneNumber\":\"1-2\"}")
                        .build(),
                AgentMessage.builder()
                        .role("assistant")
                        .reasoningContent("第一场已保存，第二场尚未完成")
                        .build(),
                AgentMessage.builder()
                        .role("assistant")
                        .parentToolCallId("sub-agent-call")
                        .content("子 Agent 内部明细")
                        .build()), "继续");

        assertThat(recovered).extracting(Msg::getRole)
                .containsExactly(
                        MsgRole.USER,
                        MsgRole.ASSISTANT,
                        MsgRole.ASSISTANT,
                        MsgRole.ASSISTANT,
                        MsgRole.USER);
        assertThat(recovered.get(1).getTextContent())
                .contains("get_script_episode", "状态：success", "第一集");
        assertThat(recovered.get(2).getTextContent())
                .contains("未完成", "save_storyboard_scene_shots", "1-2");
        assertThat(recovered.get(3).getTextContent())
                .contains("第一场已保存，第二场尚未完成")
                .doesNotContain("子 Agent 内部明细");
        assertThat(recovered.getLast().getTextContent()).isEqualTo("继续");
    }

    private AiMultimodalInputVO input(String id, String name, String inputType, String mimeType,
                                      String transport, String url, String data) {
        AiMultimodalInputVO input = new AiMultimodalInputVO();
        input.setId(id);
        input.setName(name);
        input.setInputType(inputType);
        input.setMimeType(mimeType);
        input.setTransport(transport);
        input.setUrl(url);
        input.setData(data);
        return input;
    }
}
