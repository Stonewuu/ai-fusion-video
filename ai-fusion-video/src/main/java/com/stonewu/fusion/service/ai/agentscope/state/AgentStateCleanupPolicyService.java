package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentStateCleanupPolicy;
import com.stonewu.fusion.mapper.ai.AgentStateCleanupPolicyMapper;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Objects;
import java.util.Optional;

@Service
public class AgentStateCleanupPolicyService {

    public static final long POLICY_ID = 1L;
    public static final String STATUS_ACTIVE = "ACTIVE";
    public static final String STATUS_EXPIRED = "EXPIRED";
    public static final String EXPIRED_MESSAGE =
            "该会话状态存储已过期，请发起新会话";

    private static final int MIN_CLEANUP_INTERVAL_DAYS = 1;
    private static final int MAX_CLEANUP_INTERVAL_DAYS = 365;
    private static final int MIN_RETENTION_DAYS = 1;
    private static final int MAX_RETENTION_DAYS = 3650;

    private final AgentStateCleanupPolicyMapper mapper;

    public AgentStateCleanupPolicyService(AgentStateCleanupPolicyMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper, "mapper must not be null");
    }

    @Cacheable(value = "agentStateCleanupPolicy", key = "'current'")
    public AgentStateCleanupPolicy getCurrent() {
        AgentStateCleanupPolicy policy = mapper.selectById(POLICY_ID);
        if (policy == null) {
            throw new BusinessException("AgentState 清理配置不存在，请先执行数据库迁移");
        }
        return policy;
    }

    @Transactional
    @CacheEvict(value = "agentStateCleanupPolicy", allEntries = true)
    public AgentStateCleanupPolicy update(
            int cleanupIntervalDays,
            int retentionDays) {
        requireRange(
                cleanupIntervalDays,
                MIN_CLEANUP_INTERVAL_DAYS,
                MAX_CLEANUP_INTERVAL_DAYS,
                "cleanupIntervalDays");
        requireRange(
                retentionDays,
                MIN_RETENTION_DAYS,
                MAX_RETENTION_DAYS,
                "retentionDays");
        AgentStateCleanupPolicy policy = requireLocked();
        LocalDateTime now = mapper.selectDatabaseNow();
        if (policy.getCleanupLeaseUntil() != null
                && policy.getCleanupLeaseUntil().isAfter(now)) {
            throw new BusinessException("AgentState 清理任务正在执行，请稍后重试配置");
        }
        policy.setCleanupIntervalDays(cleanupIntervalDays);
        policy.setRetentionDays(retentionDays);
        policy.setNextCleanupAt(now);
        policy.setCleanupLeaseOwner(null);
        policy.setCleanupLeaseUntil(null);
        policy.setUpdateTime(now);
        requireUpdated(policy);
        return policy;
    }

    @Transactional
    @CacheEvict(value = "agentStateCleanupPolicy", allEntries = true)
    public Optional<CleanupClaim> tryClaim(String owner, int leaseMinutes) {
        String safeOwner = requireText(owner, "owner");
        if (leaseMinutes <= 0 || leaseMinutes > 24 * 60) {
            throw new IllegalArgumentException(
                    "leaseMinutes must be between 1 and 1440");
        }
        AgentStateCleanupPolicy policy = requireLocked();
        LocalDateTime now = mapper.selectDatabaseNow();
        boolean leased = policy.getCleanupLeaseUntil() != null
                && policy.getCleanupLeaseUntil().isAfter(now);
        if (leased || policy.getNextCleanupAt().isAfter(now)) {
            return Optional.empty();
        }
        policy.setCleanupLeaseOwner(safeOwner);
        policy.setCleanupLeaseUntil(now.plusMinutes(leaseMinutes));
        policy.setUpdateTime(now);
        requireUpdated(policy);
        return Optional.of(new CleanupClaim(
                safeOwner, policy.getRetentionDays()));
    }

    @Transactional
    @CacheEvict(value = "agentStateCleanupPolicy", allEntries = true)
    public void complete(String owner) {
        AgentStateCleanupPolicy policy = requireOwnedClaim(owner);
        LocalDateTime now = mapper.selectDatabaseNow();
        policy.setLastCleanupAt(now);
        policy.setNextCleanupAt(now.plusDays(policy.getCleanupIntervalDays()));
        policy.setCleanupLeaseOwner(null);
        policy.setCleanupLeaseUntil(null);
        policy.setUpdateTime(now);
        requireUpdated(policy);
    }

    @Transactional
    @CacheEvict(value = "agentStateCleanupPolicy", allEntries = true)
    public void release(String owner) {
        AgentStateCleanupPolicy policy = requireOwnedClaim(owner);
        policy.setCleanupLeaseOwner(null);
        policy.setCleanupLeaseUntil(null);
        policy.setUpdateTime(mapper.selectDatabaseNow());
        requireUpdated(policy);
    }

    public String stateStatus(
            AgentConversation conversation,
            LocalDateTime databaseNow,
            int retentionDays) {
        return isExpired(conversation, databaseNow, retentionDays)
                ? STATUS_EXPIRED
                : STATUS_ACTIVE;
    }

    public boolean isExpired(
            AgentConversation conversation,
            LocalDateTime databaseNow,
            int retentionDays) {
        AgentConversation safeConversation = Objects.requireNonNull(
                conversation, "conversation must not be null");
        LocalDateTime safeNow = Objects.requireNonNull(
                databaseNow, "databaseNow must not be null");
        requireRange(
                retentionDays,
                MIN_RETENTION_DAYS,
                MAX_RETENTION_DAYS,
                "retentionDays");
        if (safeConversation.getAgentStateExpiredAt() != null) {
            return true;
        }
        LocalDateTime lastActive = safeConversation.getAgentStateLastActiveAt();
        return lastActive != null
                && !lastActive.plusDays(retentionDays)
                        .isAfter(safeNow);
    }

    public void requireAvailable(
            AgentConversation conversation,
            LocalDateTime databaseNow,
            int retentionDays) {
        if (isExpired(conversation, databaseNow, retentionDays)) {
            throw new BusinessException(409, EXPIRED_MESSAGE);
        }
    }

    private AgentStateCleanupPolicy requireLocked() {
        AgentStateCleanupPolicy policy = mapper.selectCurrentForUpdate();
        if (policy == null) {
            throw new BusinessException("AgentState 清理配置不存在");
        }
        return policy;
    }

    private AgentStateCleanupPolicy requireOwnedClaim(String owner) {
        String safeOwner = requireText(owner, "owner");
        AgentStateCleanupPolicy policy = requireLocked();
        if (!safeOwner.equals(policy.getCleanupLeaseOwner())) {
            throw new IllegalStateException(
                    "AgentState cleanup claim is no longer owned");
        }
        return policy;
    }

    private void requireUpdated(AgentStateCleanupPolicy policy) {
        if (mapper.updateById(policy) != 1) {
            throw new IllegalStateException(
                    "AgentState cleanup policy update failed");
        }
    }

    private void requireRange(int value, int min, int max, String field) {
        if (value < min || value > max) {
            throw new IllegalArgumentException(
                    field + " must be between " + min + " and " + max);
        }
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value.trim();
    }

    public record CleanupClaim(String owner, int retentionDays) {

        public CleanupClaim {
            if (owner == null || owner.isBlank()) {
                throw new IllegalArgumentException("owner must not be blank");
            }
            if (retentionDays < MIN_RETENTION_DAYS
                    || retentionDays > MAX_RETENTION_DAYS) {
                throw new IllegalArgumentException(
                        "retentionDays is outside the supported range");
            }
        }
    }
}
