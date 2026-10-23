package com.stonewu.fusion.service.media;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelMultimodalCapabilities;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.*;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 只为本次模型调用生成访问地址，原消息及持久化 AgentState 中继续保留稳定引用。
 */
@Component
@RequiredArgsConstructor
public class AgentMediaInputResolver {
    private final AgentRunMapper runs;
    private final MediaResourceQueries resources;
    private final MediaInputTransport transport;

    public boolean needed(List<Msg> messages) {
        return messages.stream().flatMap(message -> message.getContent().stream()).anyMatch(this::managed);
    }

    public List<Msg> resolve(List<Msg> messages, RuntimeContext runtime, AiModel model) {
        Scope scope = scope(runtime);
        Conversion converted = new Conversion();
        List<Msg> result = messages.stream().map(message -> message.withContent(message.getContent().stream()
                .map(block -> block(block, scope, model, converted)).toList())).toList();
        if (!scope.equals(scope(runtime))) {
            throw denied();
        }
        return result;
    }

    private ContentBlock block(ContentBlock block, Scope scope, AiModel model, Conversion converted) {
        if (!managed(block)) {
            return block;
        }
        return switch (block) {
            case ImageBlock image -> ImageBlock.builder()
                    .source(source(image.getSource(), scope, model, Set.of(MediaKind.IMAGE), converted))
                    .minPixels(image.getMinPixels()).maxPixels(image.getMaxPixels()).build();
            case VideoBlock video -> VideoBlock.builder()
                    .source(source(video.getSource(), scope, model, Set.of(MediaKind.VIDEO), converted))
                    .fps(video.getFps()).maxFrames(video.getMaxFrames()).minPixels(video.getMinPixels())
                    .maxPixels(video.getMaxPixels()).totalPixels(video.getTotalPixels()).build();
            case AudioBlock audio -> AudioBlock.builder()
                    .source(source(audio.getSource(), scope, model, Set.of(MediaKind.AUDIO), converted)).build();
            case DataBlock data -> DataBlock.builder().id(data.getId()).name(data.getName())
                    .source(source(data.getSource(), scope, model, Set.of(MediaKind.DOCUMENT, MediaKind.TEXT),
                            converted)).build();
            case ToolResultBlock tool -> ToolResultBlock.builder().id(tool.getId()).name(tool.getName())
                    .metadata(tool.getMetadata()).state(tool.getState())
                    .output(tool.getOutput().stream().map(child -> block(child, scope, model, converted)).toList())
                    .build();
            default -> block;
        };
    }

    private Source source(Source original, Scope scope, AiModel model, Set<MediaKind> kinds, Conversion converted) {
        if (!(original instanceof URLSource url) || !MediaReference.managed(url.getUrl())) {
            return original;
        }
        String id = MediaReference.id(url.getUrl());
        var resource = resources.get(id, scope.actor());
        if (!kinds.contains(resource.content().kind())) {
            throw MediaException.invalid("助手媒体块与资源类型不符");
        }
        if (scope.projectId() != null && resource.projectId() != null && !scope.projectId()
                .equals(resource.projectId())) {
            throw new MediaException(403, "MEDIA_AGENT_SCOPE_MISMATCH", "助手不能读取当前项目范围以外的媒体");
        }
        String method = AiModelMultimodalCapabilities.managedTransport(model, resource.content().mimeType());
        if (url.getMimeType() != null && !url.getMimeType().isBlank() && !url.getMimeType()
                .equals(resource.content().mimeType())) {
            throw MediaException.invalid("助手媒体声明与实际类型不一致");
        }
        if (++converted.blocks > 128) {
            throw MediaException.invalid("本次模型调用的媒体数量超过限制");
        }
        if (method.equals(AiModelMultimodalCapabilities.TRANSPORT_BASE64)) {
            converted.base64Bytes += resource.content().byteSize();
            if (converted.base64Bytes > 20L * 1024 * 1024) {
                throw MediaException.invalid("本次模型调用的私有 Base64 媒体总大小超过 20 MiB，请缩小观察范围");
            }
        }
        if (!converted.sources.containsKey(id)) {
            if (method.equals(AiModelMultimodalCapabilities.TRANSPORT_BASE64)) {
                String data = transport.dataUri(url.getUrl(), scope.actor(), resource.content().kind(),
                        10L * 1024 * 1024);
                converted.sources.put(id, Base64Source.builder().mediaType(resource.content().mimeType())
                        .data(data.substring(data.indexOf(',') + 1)).build());
            } else {
                converted.sources.put(id, URLSource.builder()
                        .url(transport.agentUrl(url.getUrl(), scope.actor(), resource.content().kind(), scope.run()))
                        .mimeType(resource.content().mimeType()).build());
            }
        }
        return converted.sources.get(id);
    }

    private static final class Conversion {
        final Map<String, Source> sources = new HashMap<>();
        long base64Bytes;
        int blocks;
    }

    private boolean managed(ContentBlock block) {
        return switch (block) {
            case ImageBlock image -> managed(image.getSource());
            case VideoBlock video -> managed(video.getSource());
            case AudioBlock audio -> managed(audio.getSource());
            case DataBlock data -> managed(data.getSource());
            case ToolResultBlock tool -> tool.getOutput().stream().anyMatch(this::managed);
            default -> false;
        };
    }

    private boolean managed(Source source) {
        return source instanceof URLSource url && MediaReference.managed(url.getUrl());
    }

    private Scope scope(RuntimeContext runtime) {
        if (runtime == null) {
            throw denied();
        }
        var user = runtime.get(AuthenticatedUserContext.class);
        var run = runtime.get(AgentRunContext.class);
        if (user == null || run == null || !Objects.equals(runtime.getUserId(), user.userId().toString())) {
            throw denied();
        }
        var row = runs.activeMediaScope(run.runId(), user.userId(), run.ownerInstanceId(), run.ownerEpoch());
        if (row == null) {
            throw denied();
        }
        return new Scope(new CanvasActor(user.userId(), row.getActorTeamId()), row.getProjectId(), run);
    }

    private MediaException denied() {
        return new MediaException(403, "MEDIA_AGENT_INACTIVE", "助手运行身份已失效，不能读取媒体");
    }

    private record Scope(CanvasActor actor, Long projectId, AgentRunContext run) {
    }
}
