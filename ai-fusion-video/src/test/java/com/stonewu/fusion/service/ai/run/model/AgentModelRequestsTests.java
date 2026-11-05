package com.stonewu.fusion.service.ai.run.model;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentModelRequestEntity;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.mapper.ai.AgentModelCallUsageMapper;
import com.stonewu.fusion.mapper.ai.AgentModelRequestMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.stonewu.fusion.repository.ai.AgentModelCallUsageRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.model.AiModelRequestOptions;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import io.agentscope.core.agent.RuntimeContext;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class AgentModelRequestsTests {
    private final ObjectMapper json = new ObjectMapper();
    private final AgentModelRequestMapper requests = mock(AgentModelRequestMapper.class);
    private final AgentRunMapper runs = mock(AgentRunMapper.class);
    private final AiModelMapper models = mock(AiModelMapper.class);
    private final AiProviderService providers = mock(AiProviderService.class);
    private final AgentModelCallUsageRepository usages = mock(AgentModelCallUsageRepository.class);
    private final AiModelMetadataResolver metadata = mock(AiModelMetadataResolver.class);
    private final AgentModelRequests admission = new AgentModelRequests(requests, mock(AgentModelCallUsageMapper.class),
            mock(ExecutionCoordinationMapper.class), runs, usages, mock(CanvasAccessPolicy.class),
            mock(CanvasDocumentRepository.class), new CanvasJson(json), metadata, models, providers, json);

    @BeforeAll
    static void mapperMetadata() {
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(new MybatisConfiguration(), ""), AiModel.class);
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(new MybatisConfiguration(), ""), AgentModelRequestEntity.class);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {"   ", "{}"})
    @DisplayName("附加配置未设置时仍能受理模型调用，不得报告画布数据错误")
    void emptyOptionalConfigurationCanAdmitAModelRequest(String config) {
        AiModel model = model(config);
        RuntimeContext runtime = runtime(model);
        when(providers.agentScopeModelFingerprint(any())).thenReturn("unchanged");

        assertThat(admission.admit("call", model, runtime, "unchanged")).contains(
                new AgentModelRequests.Claim("call", "run", "worker", 1));

        assertThat(model.getConfig()).isEqualTo(config);
        verify(requests).admit("call", "run", 7L, "worker", 1L);
        verify(usages).startCall("run", "call", "openai_compatible", "test-model");
    }

    @ParameterizedTest
    @ValueSource(strings = {"{broken", "[]", "null", "false"})
    @DisplayName("非空附加配置必须是合法对象，拒绝时不能登记调用或用量")
    void invalidOptionalConfigurationReportsAModelErrorBeforeAdmission(String config) {
        AiModel model = model(config);
        RuntimeContext runtime = runtime(model);
        assertThatThrownBy(() -> admission.admit("call", model, runtime, "unchanged"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("模型配置");
        verify(requests, never()).admit(anyString(), anyString(), anyLong(), anyString(), anyLong());
        verifyNoInteractions(usages, providers);
    }

    @Test
    @DisplayName("当前调用的思考等级参与配置校验，同时不覆盖其他已修改的模型参数")
    void requestEffortIsPreservedWhileOtherConfigurationChangesAreRejected() {
        AiModel persisted = model("{\"temperature\":0.2}");
        persisted.setSupportReasoning(true);
        persisted.setReasoningEffortLevels(List.of("high"));
        var runtime = runtime(persisted);
        var requested = AiModelRequestOptions.withReasoningEffort(persisted, "high", json);
        when(providers.agentScopeModelFingerprint(any())).thenAnswer(call -> ((AiModel) call.getArgument(0)).getConfig());
        assertThat(admission.admit("first", requested, runtime, requested.getConfig())).isPresent();
        assertThat(persisted.getConfig()).isEqualTo("{\"temperature\":0.2}");

        persisted.setConfig("{\"temperature\":0.7}");
        assertThatThrownBy(() -> admission.admit("second", requested, runtime, requested.getConfig()))
                .hasMessageContaining("配置在执行期间变化");
        verify(requests, never()).admit(eq("second"), anyString(), anyLong(), anyString(), anyLong());
    }

    private AiModel model(String config) {
        return AiModel.builder().id(7L).code("test-model").config(config).status(1).maxConcurrency(5).build();
    }

    private RuntimeContext runtime(AiModel model) {
        LocalDateTime now = LocalDateTime.of(2026, 11, 5, 8, 0);
        var run = AgentRun.builder().runId("run").userId(11L).status("RUNNING").ownerInstanceId("worker")
                .ownerEpoch(1L).leaseUntil(now.plusMinutes(1)).deadlineAt(now.plusMinutes(5)).build();
        when(runs.selectByRunIdForUpdate("run")).thenReturn(run);
        when(runs.selectDatabaseNow()).thenReturn(now);
        when(models.selectById(7L)).thenReturn(model);
        when(models.selectOne(any())).thenReturn(model);
        when(metadata.resolve(any())).thenReturn(new AiModelMetadata("openai_compatible", "openai_compatible", "openai_compatible"));
        return RuntimeContext.builder().userId("11").sessionId("session")
                .put(AuthenticatedUserContext.class, new AuthenticatedUserContext(11L))
                .put(AgentRunContext.class, new AgentRunContext("run", "worker", 1L, Instant.now().plusSeconds(300)))
                .build();
    }
}
