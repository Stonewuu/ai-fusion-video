package com.stonewu.fusion.service.ai.catalog;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.ai.vo.ApiConfigSaveReqVO;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.when;

class ProviderDiscoveryFailureTests {
    /** 读取余额/模型列表的失败不能暗示已经发起生成，也不能退化成无法操作的系统错误。 */
    @Test
    void reportsReadContextAndSafeNetworkGuidance() throws Exception {
        var adapter = mock(ProviderModelDiscovery.class);
        when(adapter.type()).thenReturn("metaso");
        var service = new ProviderDiscoveryService(new ProviderCatalogService(new ObjectMapper()),
                mock(ApiConfigService.class), List.of(adapter));
        var request = new ApiConfigSaveReqVO();
        request.setName("fixture");
        request.setProviderId("metaso");
        request.setApiKey("fixture-key");
        when(adapter.discover(any(), any())).thenThrow(GenerationProviderFailure.http(401,
                "{\"error\":{\"message\":\"invalid API key: sk-private-secret-123\"}}"));
        assertThatThrownBy(() -> service.preview(request)).hasMessageStartingWith("读取模型失败（HTTP 401）")
                .hasMessageNotContaining("private-secret").hasMessageNotContaining("生成失败");
        doThrow(new UncheckedIOException(new IOException("connection lost"))).when(adapter).discover(any(), any());
        assertThatThrownBy(() -> service.preview(request)).hasMessageContaining("检查 API 地址与网络连接后重试");
    }
}
