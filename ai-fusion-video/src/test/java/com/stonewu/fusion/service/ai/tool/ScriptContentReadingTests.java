package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.text.TextPages;
import com.stonewu.fusion.service.ai.tool.script.ReadScriptContentToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.ScriptToolQueries;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodePolicy;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeRepository;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeToolScope;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ScriptContentReadingTests {
    private final ObjectMapper mapper = new ObjectMapper();
    private final CanvasJson json = new CanvasJson(mapper);
    private final ResourceQueryService queries = mock(ResourceQueryService.class);
    private final ReadScriptContentToolExecutor reader = new ReadScriptContentToolExecutor(queries, json);
    private final CanvasActor actor = new CanvasActor(7, null);
    private final ToolExecutionContext context = ToolExecutionContext.forInvocation(7L, 1, 7L,
            new ResourceCommandContext(actor, "agent", null, "run", "call", "worker", 1L));

    ScriptContentReadingTests() {
        context.setModel(AiModel.builder().contextWindow(128_000).build());
    }

    @Test
    @DisplayName("查询返回可用的读取入口，长篇单行原文连续分页后逐字一致")
    void queryReferenceReadsTheWholeNovelIncludingSupplementaryUnicodeCharacters() {
        String source = "序章🌕" + "中间剧情𠮷".repeat(19_000) + "终章";
        ObjectNode data = mapper.createObjectNode().put("id", 10).put("version", 2).put("rawContent", source);
        when(queries.get(CanvasValueType.SCRIPT, 10, actor))
                .thenAnswer(ignored -> new ResourceSnapshot(null, data.deepCopy()));
        String query = new ScriptToolQueries(queries, json).execute("get_script", "{\"scriptId\":10}", context);
        ObjectNode result = json.read(query, ObjectNode.class);
        assertThat(result.has("rawContent")).isFalse();
        ObjectNode request = (ObjectNode) result.path("contentReads").path("rawContent").deepCopy();
        assertThat(request.remove("readTool").asText()).isEqualTo("read_script_content");
        request.remove("totalChars");
        request.put("limit", 7999);
        StringBuilder complete = new StringBuilder();
        int previousEnd = 0;
        while (true) {
            ObjectNode page = json.read(reader.execute(json.write(request), context), ObjectNode.class);
            assertThat(page.path("status").asText()).isEqualTo("success");
            assertThat(page.path("offset").asInt()).isEqualTo(previousEnd);
            String content = page.path("content").asText();
            assertThat(content.codePointCount(0, content.length())).isLessThanOrEqualTo(7999);
            assertThat(Character.isLowSurrogate(content.charAt(0))).isFalse();
            assertThat(Character.isHighSurrogate(content.charAt(content.length() - 1))).isFalse();
            complete.append(content);
            previousEnd = page.path("endOffset").asInt();
            if (!page.path("hasMore").asBoolean()) {
                assertThat(page.path("nextOffset").isNull()).isTrue();
                assertThat(previousEnd).isEqualTo(page.path("totalChars").asInt());
                break;
            }
            request.set("offset", page.get("nextOffset"));
        }
        assertThat(complete.toString()).isEqualTo(source);
    }

    @Test
    @DisplayName("分集摘要仍提供原文入口，父级或场次版本更新不误判为正文变化")
    void episodeSummaryKeepsTheReadReferenceAndOnlyTextChangesInvalidateIt() {
        ObjectNode data = mapper.createObjectNode().put("id", 12).put("scriptId", 10).put("episodeNumber", 1)
                .put("version", 3).put("rawContent", "第一场\n第二场🌕");
        data.putArray("scenes");
        when(queries.scriptTree(CanvasValueType.SCRIPT_EPISODE, 12, actor)).thenReturn(data.deepCopy());
        var summary = json.read(new ScriptToolQueries(queries, json).execute("get_script_episode",
                "{\"scriptEpisodeId\":12}", context), ObjectNode.class);
        var entry = summary.path("contentReads").path("rawContent");
        assertThat(entry.path("resourceType").asText()).isEqualTo("episode");
        ObjectNode request = mapper.createObjectNode().put("resourceType", "episode").put("resourceId", 12)
                .put("field", "rawContent").put("offset", 3).put("contentVersion", entry.path("contentVersion").asText());
        data.put("version", 4);
        when(queries.get(CanvasValueType.SCRIPT_EPISODE, 12, actor)).thenReturn(new ResourceSnapshot(null, data));
        assertThat(json.read(reader.execute(json.write(request), context), ObjectNode.class).path("status").asText())
                .isEqualTo("success");
        data.put("rawContent", "原文已由其他人修改");
        assertThat(json.read(reader.execute(json.write(request), context), ObjectNode.class).path("message").asText())
                .contains("原文已变化");
    }

    @Test
    @DisplayName("继续读取时重新检查权限，已读取一页不代表之后仍有访问权限")
    void revokedAccessStopsTheNextPage() {
        ObjectNode data = mapper.createObjectNode().put("rawContent", "原文内容").put("version", 1);
        when(queries.get(CanvasValueType.SCRIPT, 10, actor)).thenReturn(new ResourceSnapshot(null, data))
                .thenThrow(new BusinessException(403, "无权读取剧本"));
        ObjectNode request = json.object(Map.of("resourceType", "script", "resourceId", 10, "field", "rawContent",
                "limit", 2));
        var page = json.read(reader.execute(json.write(request), context), ObjectNode.class);
        request.set("offset", page.get("nextOffset"));
        request.set("contentVersion", page.get("contentVersion"));
        assertThatThrownBy(() -> reader.execute(json.write(request), context))
                .isInstanceOf(BusinessException.class).hasMessage("无权读取剧本");
    }

    @Test
    @DisplayName("画布中的分段读取不能借通用业务 ID 越过当前项目范围")
    void canvasReaderKeepsTheCurrentProjectBoundary() {
        var policy = mock(CanvasAgentNodePolicy.class);
        when(policy.job(context.requireMutationContext(), false)).thenReturn(mock(CanvasAgentNodeRepository.Job.class));
        var scope = mock(CanvasAgentScope.class);
        when(scope.nodeRunId()).thenReturn("node-run");
        when(scope.projectId()).thenReturn(100L);
        context.setCanvasScope(scope);
        var guard = new CanvasAgentNodeToolScope(policy, queries, json, mock(GenerationSubmissionMapper.class));
        when(queries.get(CanvasValueType.SCRIPT, 10, actor)).thenReturn(new ResourceSnapshot(
                new ResourceHeader(CanvasValueType.SCRIPT, 10, 200L, 1, false), mapper.createObjectNode()));
        Map<String, Object> input = Map.of("resourceType", "script", "resourceId", 10, "field", "rawContent");
        assertThatThrownBy(() -> guard.before("read_script_content", input, context))
                .hasMessageContaining("不属于当前画布项目");
        when(queries.get(CanvasValueType.SCRIPT, 10, actor)).thenReturn(new ResourceSnapshot(
                new ResourceHeader(CanvasValueType.SCRIPT, 10, 100L, 1, false), mapper.createObjectNode()));
        assertThatCode(() -> guard.before("read_script_content", input, context)).doesNotThrowAnyException();
    }

    @ParameterizedTest
    @ValueSource(strings = {"{\"offset\":-1}", "{\"limit\":0}", "{\"limit\":64001}",
            "{\"offset\":1}", "{\"offset\":0.5}", "{\"limit\":\"4000\"}"})
    @DisplayName("非法范围和没有内容版本的续读请求不得默认为成功")
    void invalidPaginationIsRejected(String input) {
        assertThatThrownBy(() -> TextPages.Request.from(json.read(input, ObjectNode.class), 128_000))
                .isInstanceOf(BusinessException.class);
    }

    @Test
    @DisplayName("大容量模型减少长正文往返，小容量模型限制单次读取并保留完整续读位置")
    void readingBudgetFollowsTheConfiguredModelAndKeepsExplicitSmallerRequests() {
        String source = "剧情🌕".repeat(30_000);
        when(queries.get(CanvasValueType.SCRIPT, 10, actor)).thenReturn(new ResourceSnapshot(null,
                mapper.createObjectNode().put("rawContent", source).put("version", 1)));
        String request = "{\"resourceType\":\"script\",\"resourceId\":10,\"field\":\"rawContent\"}";
        context.setModel(AiModel.builder().contextWindow(1_000_000).build());
        var large = json.read(reader.execute(request, context), ObjectNode.class);
        assertThat(large.path("endOffset").intValue()).isEqualTo(64_000);
        assertThat(large.path("nextOffset").intValue()).isEqualTo(64_000);
        assertThat(large.path("totalChars").intValue()).isEqualTo(90_000);
        assertThat(large.path("hasMore").booleanValue()).isTrue();
        context.setModel(AiModel.builder().contextWindow(32_000).build());
        var bounded = json.read(reader.execute(request.replace("}", ",\"limit\":64000}"), context), ObjectNode.class);
        assertThat(bounded.path("endOffset").intValue()).isEqualTo(4_000);
        var shortRead = json.read(reader.execute(request.replace("}", ",\"limit\":1000}"), context), ObjectNode.class);
        assertThat(shortRead.path("endOffset").intValue()).isEqualTo(1_000);
    }

    @Test
    @DisplayName("空正文和末尾读取明确结束，越界位置报错")
    void emptyAndEndOfContentAreUnambiguous() {
        var empty = TextPages.read("", new TextPages.Request(0, 10, null));
        assertThat(empty.content()).isEmpty();
        assertThat(empty.hasMore()).isFalse();
        assertThat(empty.nextOffset()).isNull();
        var last = TextPages.read("结尾", new TextPages.Request(2, 10, TextPages.version("结尾")));
        assertThat(last.content()).isEmpty();
        assertThatThrownBy(() -> TextPages.read("结尾", new TextPages.Request(3, 10, TextPages.version("结尾"))))
                .hasMessageContaining("超出");
    }
}
