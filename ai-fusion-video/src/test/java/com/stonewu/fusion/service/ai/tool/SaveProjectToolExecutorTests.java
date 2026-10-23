package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.project.SaveProjectToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class SaveProjectToolExecutorTests {
    private final ResourceCommandService commands = mock(ResourceCommandService.class);
    private final CanvasJson json = new CanvasJson(new ObjectMapper());
    private final SaveProjectToolExecutor tool = new SaveProjectToolExecutor(commands, json);
    private final ResourceCommandContext mutation = new ResourceCommandContext(new CanvasActor(42, 7L), "agent", null, "run", "call", "owner", 1L);
    private final ToolExecutionContext context = ToolExecutionContext.forInvocation(42L, 2, 7L, mutation);

    @Test
    void requiresObservedVersionAndTrustedContextBeforeAnyWrite() {
        assertThat(tool.execute("{\"projectId\":8,\"name\":\"猜测版本\"}", context)).contains("expectedVersion");
        assertThat(tool.execute("{\"projectId\":8,\"expectedVersion\":0,\"name\":\"缺少上下文\"}",
                ToolExecutionContext.builder().userId(42L).build())).contains("可信");
        verifyNoInteractions(commands);
    }

    @Test
    void preservesCallIdentityVersionNullAndStructuredPropertiesAcrossRetry() {
        when(commands.execute(any(), eq(mutation))).thenReturn(new ResourceCommandReceipt("committed", "request",
                JsonNodeFactory.instance.objectNode().put("version", 4), List.of()));
        String input = "{\"projectId\":8,\"expectedVersion\":3,\"artStyle\":null,\"properties\":\"{\\\"aspectRatio\\\":\\\"16:9\\\"}\"}";
        assertThat(tool.execute(input, context)).contains("committed", "changes");
        tool.execute(input, context);
        ArgumentCaptor<ResourcePatchCommand> calls = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands, times(2)).execute(calls.capture(), eq(mutation));
        assertThat(calls.getAllValues().get(0).requestId()).isEqualTo(calls.getAllValues().get(1).requestId());
        assertThat(calls.getValue().expectedVersion()).isEqualTo(3);
        assertThat(calls.getValue().patch().path("artStyle").isNull()).isTrue();
        assertThat(calls.getValue().patch().path("properties").isObject()).isTrue();
    }
}
