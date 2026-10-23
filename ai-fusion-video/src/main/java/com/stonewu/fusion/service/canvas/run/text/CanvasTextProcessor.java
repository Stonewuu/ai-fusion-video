package com.stonewu.fusion.service.canvas.run.text;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.AiModelMultimodalCapabilities;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.provider.SingleRequestModelSupport;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaKind;
import io.agentscope.core.message.*;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ChatUsage;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.*;

import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Job;

@Component
@RequiredArgsConstructor
public class CanvasTextProcessor {
    private final MediaResourceQueries media;
    private final MediaInputTransport transport;
    private final CanvasRunSources sources;
    private final CanvasJson json;

    public List<Msg> messages(Job job, Run run) {
        var input = job.input();
        if (input.schemaVersion() != 1) {
            throw CanvasRunException.invalid("不支持的文本任务快照版本");
        }
        input.inputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
        var images = input.inputs().at("images");
        if (images.size() > 8) {
            throw CanvasRunException.invalid("文本模型单次最多接收 8 张图片");
        }
        List<ContentBlock> content = new ArrayList<>();
        content.add(TextBlock.builder().text(input.inputs().at("prompt").getFirst().text()).build());
        long bytes = 0;
        for (var value : images) {
            var resource = media.get(value.mediaId(), run.actor());
            var supported = input.model().getMultimodalInputTransports().get("image");
            String transfer = supported != null && supported.contains("base64") ? "base64" : "url";
            AiModelMultimodalCapabilities.validateUpload(input.model(), resource.content().mimeType(), transfer);
            if (transfer.equals("base64")) {
                bytes += resource.content().byteSize();
                if (bytes > 20 * 1024 * 1024) {
                    throw CanvasRunException.invalid("文本模型的图片总大小不能超过 20 MiB");
                }
                String data = transport.dataUri(MediaReference.of(value.mediaId()), run.actor(), MediaKind.IMAGE,
                        10 * 1024 * 1024);
                content.add(ImageBlock.builder().source(Base64Source.builder().mediaType(resource.content().mimeType())
                        .data(data.substring(data.indexOf(',') + 1)).build()).build());
            } else {
                content.add(ImageBlock.builder().source(URLSource.builder().mimeType(resource.content().mimeType())
                        .url(transport.canvasTextUrl(MediaReference.of(value.mediaId()), run.actor(), job.claim()))
                        .build()).build());
            }
        }
        List<Msg> messages = new ArrayList<>();
        if (!input.instruction().isBlank()) {
            messages.add(Msg.builder().name("system").role(MsgRole.SYSTEM)
                    .content(TextBlock.builder().text(input.instruction()).build()).build());
        }
        messages.add(Msg.builder().name("user").role(MsgRole.USER).content(content).build());
        return List.copyOf(messages);
    }

    public ObjectNode generate(OwnedChatModel model, List<Msg> messages) {
        var result = model.model().stream(messages, List.of(), SingleRequestModelSupport.options(null))
                .reduce(new Result(), Result::add).block(Duration.ofMinutes(25));
        if (result == null || result.finish == null || result.finish.isBlank()) {
            throw CanvasRunException.invalid("模型连接结束但没有完整终态，请核实原请求");
        }
        var output = json.object(Map.of("text", result.text.toString(), "finishReason", result.finish,
                "complete", !result.truncated && !result.toolCall && !result.text.isEmpty()
                        && Set.of("stop", "end_turn", "stop_sequence")
                        .contains(result.finish.toLowerCase(Locale.ROOT))));
        if (result.usage != null) {
            output.set("usage", json.object(result.usage));
        }
        return output;
    }

    private static final class Result {
        private final StringBuilder text = new StringBuilder();
        private String finish;
        private ChatUsage usage;
        private boolean truncated;
        private boolean toolCall;

        Result add(ChatResponse response) {
            for (var block : response.getContent()) {
                if (block instanceof TextBlock value && value.getText() != null) {
                    int remaining = 128000 - text.length();
                    truncated |= value.getText().length() > remaining;
                    text.append(value.getText(), 0, Math.min(remaining, value.getText().length()));
                } else if (block instanceof ToolUseBlock) {
                    toolCall = true;
                }
            }
            if (response.getFinishReason() != null) {
                finish = response.getFinishReason();
            }
            if (response.getUsage() != null) {
                usage = response.getUsage();
            }
            return this;
        }
    }
}
