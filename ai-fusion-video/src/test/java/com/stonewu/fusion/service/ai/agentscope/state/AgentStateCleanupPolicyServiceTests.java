package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentStateCleanupPolicy;
import com.stonewu.fusion.mapper.ai.AgentStateCleanupPolicyMapper;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentStateCleanupPolicyServiceTests {

    private final AgentStateCleanupPolicyMapper mapper =
            mock(AgentStateCleanupPolicyMapper.class);
    private final AgentStateCleanupPolicyService service =
            new AgentStateCleanupPolicyService(mapper);

    @Test
    void expirationUsesTheConfiguredRetentionBoundaryAndPhysicalTombstone() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 30, 12, 0);
        AgentConversation recent = AgentConversation.builder()
                .agentStateLastActiveAt(now.minusDays(29))
                .build();
        AgentConversation boundary = AgentConversation.builder()
                .agentStateLastActiveAt(now.minusDays(30))
                .build();
        AgentConversation physicallyExpired = AgentConversation.builder()
                .agentStateLastActiveAt(now.minusDays(1))
                .agentStateExpiredAt(now.minusHours(1))
                .build();

        assertThat(service.stateStatus(recent, now, 30))
                .isEqualTo(AgentStateCleanupPolicyService.STATUS_ACTIVE);
        assertThat(service.stateStatus(boundary, now, 30))
                .isEqualTo(AgentStateCleanupPolicyService.STATUS_EXPIRED);
        assertThat(service.isExpired(physicallyExpired, now, 30)).isTrue();
        assertThatThrownBy(() -> service.requireAvailable(boundary, now, 30))
                .isInstanceOfSatisfying(BusinessException.class, failure -> {
                    assertThat(failure.getCode()).isEqualTo(409);
                    assertThat(failure.getMessage()).isEqualTo(
                            AgentStateCleanupPolicyService.EXPIRED_MESSAGE);
                });
    }

    @Test
    void updatePersistsTheUserPolicyAndMakesItDueForEvaluation() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 30, 12, 0);
        AgentStateCleanupPolicy policy = policy(now.plusDays(1));
        when(mapper.selectCurrentForUpdate()).thenReturn(policy);
        when(mapper.selectDatabaseNow()).thenReturn(now);
        when(mapper.updateById(policy)).thenReturn(1);

        AgentStateCleanupPolicy updated = service.update(7, 90);

        assertThat(updated.getCleanupIntervalDays()).isEqualTo(7);
        assertThat(updated.getRetentionDays()).isEqualTo(90);
        assertThat(updated.getNextCleanupAt()).isEqualTo(now);
        verify(mapper).updateById(policy);
    }

    @Test
    void distributedClaimSchedulesTheNextRunOnlyAfterSuccessfulCompletion() {
        LocalDateTime startedAt = LocalDateTime.of(2026, 7, 30, 12, 0);
        LocalDateTime completedAt = startedAt.plusMinutes(3);
        AgentStateCleanupPolicy policy = policy(startedAt.minusMinutes(1));
        policy.setCleanupIntervalDays(5);
        policy.setRetentionDays(60);
        when(mapper.selectCurrentForUpdate()).thenReturn(policy);
        when(mapper.selectDatabaseNow()).thenReturn(startedAt, completedAt);
        when(mapper.updateById(policy)).thenReturn(1);

        Optional<AgentStateCleanupPolicyService.CleanupClaim> claim =
                service.tryClaim("node-a:job-1", 120);
        assertThat(claim).contains(new AgentStateCleanupPolicyService.CleanupClaim(
                "node-a:job-1", 60));
        assertThat(policy.getCleanupLeaseUntil()).isEqualTo(
                startedAt.plusMinutes(120));

        service.complete("node-a:job-1");

        assertThat(policy.getCleanupLeaseOwner()).isNull();
        assertThat(policy.getLastCleanupAt()).isEqualTo(completedAt);
        assertThat(policy.getNextCleanupAt()).isEqualTo(completedAt.plusDays(5));
    }

    private AgentStateCleanupPolicy policy(LocalDateTime nextCleanupAt) {
        return AgentStateCleanupPolicy.builder()
                .id(AgentStateCleanupPolicyService.POLICY_ID)
                .cleanupIntervalDays(1)
                .retentionDays(30)
                .nextCleanupAt(nextCleanupAt)
                .build();
    }
}
