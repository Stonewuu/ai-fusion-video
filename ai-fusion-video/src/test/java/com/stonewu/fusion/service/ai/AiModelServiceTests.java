package com.stonewu.fusion.service.ai;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiModelConnectivityRespVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.BeforeEach;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AiModelServiceTests {

    @BeforeAll
    static void initMybatisPlusTableInfo() {
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(new MybatisConfiguration(), ""), AiModel.class);
    }

    @Mock
    private AiModelMapper aiModelMapper;

    @Mock
    private ApiConfigService apiConfigService;

    @Mock
    private ModelPresetService modelPresetService;

    @Mock
    private ChatModelFactory chatModelFactory;

    private AiModelService aiModelService;

    @BeforeEach
    void setUp() {
        aiModelService = new AiModelService(
                aiModelMapper,
                apiConfigService,
                modelPresetService,
                new AiModelMetadataResolver(apiConfigService),
                chatModelFactory
        );
    }

    @Test
    void createAiModelShouldClearOtherDefaultsWhenSavedAsDefault() {
        AiModel model = AiModel.builder()
                .name("Flux Kontext")
                .code("flux-kontext")
                .modelType(2)
                .apiConfigId(1L)
                .defaultModel(true)
                .build();
        when(apiConfigService.getById(1L)).thenReturn(ApiConfig.builder()
                .id(1L)
                .name("OpenAI Compatible")
                .imageProtocol("openai")
                .build());
        doAnswer(invocation -> {
            AiModel inserted = invocation.getArgument(0);
            inserted.setId(101L);
            return 1;
        }).when(aiModelMapper).insert(model);

        Long id = aiModelService.createAiModel(model);

        assertEquals(101L, id);
        verifyClearOtherDefaults(2, 101L);
    }

    @Test
    void updateAiModelShouldClearOtherDefaultsWhenMarkedAsDefault() {
        AiModel existing = AiModel.builder()
                .id(202L)
                .name("Wan 2.2")
                .code("wan-2.2")
                .modelType(3)
                .apiConfigId(1L)
                .defaultModel(false)
                .build();
        when(aiModelMapper.selectById(202L)).thenReturn(existing);
        when(apiConfigService.getById(1L)).thenReturn(ApiConfig.builder()
                .id(1L)
                .videoProtocol("dashscope")
                .build());

        aiModelService.updateAiModel(
                202L,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                true,
                null,
                null,
                null,
                null,
                null
        );

        assertTrue(existing.getDefaultModel());
        verify(aiModelMapper).updateById(existing);
        verify(chatModelFactory).evict(202L);
        verifyClearOtherDefaults(3, 202L);
    }

    @Test
    void testTextModelConnectivityShouldReturnResponseText() {
        AiModel model = AiModel.builder()
                .id(303L)
                .name("Gemini Flash")
                .code("gemini-2.5-flash")
                .modelType(1)
                .build();
        ChatModel chatModel = mock(ChatModel.class);
        ChatResponse chatResponse = new ChatResponse(
                java.util.List.of(new Generation(new AssistantMessage("OK")))
        );

        when(aiModelMapper.selectById(303L)).thenReturn(model);
        when(chatModelFactory.getOrCreate(model)).thenReturn(chatModel);
        when(chatModel.call(any(Prompt.class))).thenReturn(chatResponse);

        AiModelConnectivityRespVO result = aiModelService.testTextModelConnectivity(303L);

        assertEquals(303L, result.getModelId());
        assertEquals("Gemini Flash", result.getModelName());
        assertEquals("OK", result.getResponseText());
        assertThat(result.getDurationMs()).isNotNull().isGreaterThanOrEqualTo(0L);
        assertThat(result.getTestedAt()).isNotNull();
    }

    @Test
    void testTextModelConnectivityShouldRejectNonTextModel() {
        AiModel model = AiModel.builder()
                .id(304L)
                .name("Flux Kontext")
                .code("flux-kontext")
                .modelType(2)
                .build();
        when(aiModelMapper.selectById(304L)).thenReturn(model);

        BusinessException exception = assertThrows(BusinessException.class,
                () -> aiModelService.testTextModelConnectivity(304L));

        assertEquals(400, exception.getCode());
        assertEquals("仅支持文本模型连通性检测", exception.getMessage());
    }

    @Test
    void createAiModelShouldNotSelectCapabilityPresetFromModelCode() {
        AiModel model = AiModel.builder()
                .name("GPT Image 1")
                .code("gpt-image-1")
                .modelProtocol("openai")
                .modelType(2)
                .apiConfigId(1L)
                .build();
        when(apiConfigService.getById(1L)).thenReturn(ApiConfig.builder()
                .id(1L)
                .platform("openai_compatible")
                .imageProtocol("openai")
                .build());
        doAnswer(invocation -> {
            AiModel inserted = invocation.getArgument(0);
            inserted.setId(500L);
            return 1;
        }).when(aiModelMapper).insert(model);

        aiModelService.createAiModel(model);

        assertNull(model.getCapabilityPresetCode());
    }

        @Test
        void createAiModelShouldNormalizeExplicitMetadata() {
                AiModel model = AiModel.builder()
                                .name("Kling via NewAPI")
                                .code("kling-v1")
                                .capabilityPresetCode(" agnes-video-v2.0 ")
                                .modelProtocol(" Sora ")
                                .modelType(3)
                                .apiConfigId(1L)
                                .build();

                when(apiConfigService.getById(1L)).thenReturn(ApiConfig.builder().id(1L).platform("newapi").build());
                when(modelPresetService.getPreset("agnes-video-v2.0")).thenReturn(JSONUtil.createObj()
                                .set("modelType", 3)
                                .set("modelProtocol", "sora"));
                doAnswer(invocation -> {
                        AiModel inserted = invocation.getArgument(0);
                        inserted.setId(501L);
                        return 1;
                }).when(aiModelMapper).insert(model);

                aiModelService.createAiModel(model);

                assertEquals("agnes-video-v2.0", model.getCapabilityPresetCode());
                assertEquals("sora", model.getModelProtocol());
        }

    private void verifyClearOtherDefaults(int expectedModelType, long excludeId) {
        @SuppressWarnings("rawtypes")
        ArgumentCaptor<LambdaUpdateWrapper> wrapperCaptor = ArgumentCaptor.forClass(LambdaUpdateWrapper.class);

        verify(aiModelMapper).update(isNull(), wrapperCaptor.capture());

        LambdaUpdateWrapper<?> wrapper = wrapperCaptor.getValue();
        assertThat(wrapper.getSqlSet()).contains("default_model");
        assertThat(wrapper.getSqlSegment()).contains("default_model")
                .contains("model_type")
                .contains("id");
        assertThat(wrapper.getParamNameValuePairs().values())
                .contains(expectedModelType, excludeId, true, false);
    }
}
