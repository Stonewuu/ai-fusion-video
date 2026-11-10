package com.stonewu.fusion.service.ai.run;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.classic.spi.IThrowableProxy;
import ch.qos.logback.core.read.ListAppender;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import reactor.core.Exceptions;
import reactor.core.publisher.Mono;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AgentRunMaintenanceSchedulerTests {
    /** 多条恢复失败合并后，日志仍保留每条异常，不能只剩 CompositeException 名称。 */
    @Test
    void logsAllUnderlyingFailuresOfARecoveryBatch() {
        var executions = mock(OwnedExecutionRegistry.class);
        var reconciliation = mock(AgentRunReconciliationService.class);
        var cancellations = mock(CancellationCoordinator.class);
        var projections = mock(AgentMessageProjectionService.class);
        when(executions.snapshot()).thenReturn(List.of());
        when(reconciliation.reconcileBatch(anyInt())).thenReturn(Mono.empty());
        when(cancellations.retryBatch(anyInt())).thenReturn(Mono.empty());
        when(projections.recoverTerminalBatch(anyInt())).thenReturn(Mono.error(Exceptions.multiple(
                new IllegalStateException("会话状态更新失败: runId=run-a"),
                new IllegalStateException("消息顺序冲突: runId=run-b"))));
        var scheduler = new AgentRunMaintenanceScheduler(executions, mock(RunLeaseGuard.class),
                reconciliation, cancellations, projections, new AgentScopeV2Properties());
        var logger = (Logger) LoggerFactory.getLogger(AgentRunMaintenanceScheduler.class);
        var output = new ListAppender<ILoggingEvent>();
        output.start();
        logger.addAppender(output);
        try {
            scheduler.maintain();
            assertThat(output.list).singleElement().satisfies(event -> {
                assertThat(event.getThrowableProxy()).isNotNull();
                assertThat(event.getThrowableProxy().getSuppressed()).extracting(IThrowableProxy::getMessage)
                        .containsExactly("会话状态更新失败: runId=run-a", "消息顺序冲突: runId=run-b");
            });
        } finally {
            logger.detachAppender(output);
            output.stop();
        }
    }
}
