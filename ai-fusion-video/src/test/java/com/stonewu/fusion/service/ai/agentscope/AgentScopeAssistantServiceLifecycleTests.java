package com.stonewu.fusion.service.ai.agentscope;

import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.AiStreamRedisService;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import org.springframework.test.util.ReflectionTestUtils;
import reactor.core.Disposable;
import reactor.core.Disposables;

import java.util.concurrent.ConcurrentHashMap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentScopeAssistantServiceLifecycleTests {

    @Test
    @SuppressWarnings("unchecked")
    void cancelDisposesEveryActiveInvocationForTheConversationOnly() {
        StringRedisTemplate redis = mock(StringRedisTemplate.class);
        ValueOperations<String, String> values = mock(ValueOperations.class);
        when(redis.opsForValue()).thenReturn(values);
        AiStreamRedisService streamRedis = mock(AiStreamRedisService.class);
        when(streamRedis.publish(anyString(), any())).thenReturn(null);
        AgentScopeAssistantService service = new AgentScopeAssistantService(
                mock(AiModelService.class),
                mock(AiAgentService.class),
                mock(AiToolConfigService.class),
                mock(AgentConversationService.class),
                mock(AgentMessageService.class),
                mock(AgentScopeModelFactory.class),
                redis,
                streamRedis);
        ConcurrentHashMap<String, Disposable> agentCalls =
                (ConcurrentHashMap<String, Disposable>) ReflectionTestUtils.getField(
                        service, "agentCallSubscriptions");
        ConcurrentHashMap<String, Disposable> subscriptions =
                (ConcurrentHashMap<String, Disposable>) ReflectionTestUtils.getField(
                        service, "activeSubscriptions");
        Disposable firstAgent = mock(Disposable.class);
        Disposable secondAgent = mock(Disposable.class);
        Disposable otherAgent = mock(Disposable.class);
        Disposable firstSubscription = mock(Disposable.class);
        Disposable secondSubscription = mock(Disposable.class);
        Disposable otherSubscription = mock(Disposable.class);
        Disposable.Swap lateAgentHandle = Disposables.swap();
        Disposable lateAgent = mock(Disposable.class);
        String prefix = "conversation-1".length() + ":conversation-1:";
        agentCalls.put(prefix + "message-a", firstAgent);
        agentCalls.put(prefix + "message-b", secondAgent);
        agentCalls.put(prefix + "message-late", lateAgentHandle);
        agentCalls.put("14:conversation-2:message-c", otherAgent);
        subscriptions.put(prefix + "message-a", firstSubscription);
        subscriptions.put(prefix + "message-b", secondSubscription);
        subscriptions.put("14:conversation-2:message-c", otherSubscription);

        service.cancelStream("conversation-1");
        lateAgentHandle.update(lateAgent);

        verify(firstAgent).dispose();
        verify(secondAgent).dispose();
        verify(lateAgent).dispose();
        verify(firstSubscription).dispose();
        verify(secondSubscription).dispose();
        verify(otherAgent, never()).dispose();
        verify(otherSubscription, never()).dispose();
        assertThat(agentCalls).containsOnlyKeys("14:conversation-2:message-c");
        assertThat(subscriptions).containsOnlyKeys("14:conversation-2:message-c");
    }
}
