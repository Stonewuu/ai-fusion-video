package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowMapper;
import com.stonewu.fusion.mapper.ai.ComfyUiWorkflowVersionMapper;
import com.stonewu.fusion.service.ai.ApiConfigService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ComfyUiWorkflowServiceTests {

    @Mock
    private ComfyUiWorkflowMapper workflowMapper;
    @Mock
    private ComfyUiWorkflowVersionMapper versionMapper;
    @Mock
    private AiModelMapper aiModelMapper;
    @Mock
    private ApiConfigService apiConfigService;

    private ComfyUiWorkflowService service;

    @BeforeEach
    void setUp() {
        service = new ComfyUiWorkflowService(
                workflowMapper,
                versionMapper,
                aiModelMapper,
                apiConfigService,
                new ComfyUiWorkflowDocumentService(new ObjectMapper()));
    }

    @Test
    void createWorkflowNormalizesStableCodeAndDefaultsStatus() {
        when(apiConfigService.getById(7L)).thenReturn(ApiConfig.builder()
                .id(7L)
                .platform("comfyui")
                .build());
        doAnswer(invocation -> {
            ComfyUiWorkflow workflow = invocation.getArgument(0);
            workflow.setId(11L);
            return 1;
        }).when(workflowMapper).insert(any(ComfyUiWorkflow.class));

        Long id = service.createWorkflow(7L, " SDXL ", " SDXL Main ", 2, " test ", null);

        assertThat(id).isEqualTo(11L);
        ArgumentCaptor<ComfyUiWorkflow> captor = ArgumentCaptor.forClass(ComfyUiWorkflow.class);
        verify(workflowMapper).insert(captor.capture());
        assertThat(captor.getValue().getName()).isEqualTo("SDXL");
        assertThat(captor.getValue().getCode()).isEqualTo("sdxl-main");
        assertThat(captor.getValue().getStatus()).isEqualTo(1);
    }

    @Test
    void updateWorkflowCannotMoveDraftToAnotherComfyUiProvider() {
        ComfyUiWorkflow workflow = ComfyUiWorkflow.builder()
                .id(12L)
                .apiConfigId(7L)
                .modelType(2)
                .build();
        when(workflowMapper.selectById(12L)).thenReturn(workflow);

        assertThatThrownBy(() -> service.updateWorkflow(
                12L, 8L, null, null, null, null, null))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("不能更换所属供应商");
    }

    @Test
    void publishRequiresOnlineValidationAndRealExecutionTest() {
        ComfyUiWorkflow workflow = ComfyUiWorkflow.builder().id(12L).modelType(2).build();
        ComfyUiWorkflowVersion version = ComfyUiWorkflowVersion.builder()
                .id(21L)
                .workflowId(12L)
                .validationStatus(ComfyUiWorkflowVersion.VALIDATION_VALID)
                .testStatus(ComfyUiWorkflowVersion.TEST_FAILED)
                .published(false)
                .build();
        when(workflowMapper.selectOne(any())).thenReturn(workflow);
        when(versionMapper.selectById(21L)).thenReturn(version);

        assertThatThrownBy(() -> service.publishVersion(12L, 21L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("试运行成功");
    }

    @Test
    void publishMarksImmutableVersionAndSwitchesActiveVersion() {
        ComfyUiWorkflow workflow = ComfyUiWorkflow.builder().id(12L).modelType(2).status(0).build();
        ComfyUiWorkflowVersion version = ComfyUiWorkflowVersion.builder()
                .id(21L)
                .workflowId(12L)
                .validationStatus(ComfyUiWorkflowVersion.VALIDATION_VALID)
                .testStatus(ComfyUiWorkflowVersion.TEST_PASSED)
                .published(false)
                .build();
        when(workflowMapper.selectOne(any())).thenReturn(workflow);
        when(versionMapper.selectById(21L)).thenReturn(version);

        service.publishVersion(12L, 21L);

        assertThat(version.getPublished()).isTrue();
        assertThat(workflow.getActiveVersionId()).isEqualTo(21L);
        assertThat(workflow.getStatus()).isEqualTo(1);
        verify(versionMapper).updateById(version);
        verify(workflowMapper).updateById(workflow);
    }

    @Test
    void validateModelBindingRequiresSamePublishedWorkflowAndApiConfig() {
        ApiConfig apiConfig = ApiConfig.builder()
                .id(7L)
                .platform("comfyui")
                .imageProtocol("comfyui")
                .videoProtocol("comfyui")
                .build();
        ComfyUiWorkflow workflow = ComfyUiWorkflow.builder()
                .id(12L)
                .apiConfigId(7L)
                .modelType(2)
                .activeVersionId(21L)
                .status(1)
                .build();
        when(workflowMapper.selectById(12L)).thenReturn(workflow);
        AiModel model = AiModel.builder()
                .modelType(2)
                .apiConfigId(7L)
                .comfyuiWorkflowId(12L)
                .build();

        assertThatCode(() -> service.validateModelBinding(model, apiConfig)).doesNotThrowAnyException();

        model.setModelType(3);
        assertThatThrownBy(() -> service.validateModelBinding(model, apiConfig))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("模型类型");
    }
}
