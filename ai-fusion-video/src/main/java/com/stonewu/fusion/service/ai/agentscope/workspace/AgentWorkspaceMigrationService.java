package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentWorkspaceConfig;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.entity.ai.AgentWorkspaceMigration;
import com.stonewu.fusion.entity.ai.AgentWorkspaceMigrationItem;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceMigrationItemMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceMigrationMapper;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.core.task.TaskExecutor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;

@Service
public class AgentWorkspaceMigrationService {

    private final AgentWorkspaceConfigService configService;
    private final AgentWorkspacePayloadService payloadService;
    private final AgentWorkspaceCutoverService cutoverService;
    private final AgentWorkspaceEntryMapper entryMapper;
    private final AgentWorkspaceMigrationMapper migrationMapper;
    private final AgentWorkspaceMigrationItemMapper itemMapper;
    private final TaskExecutor migrationExecutor;

    public AgentWorkspaceMigrationService(
            AgentWorkspaceConfigService configService,
            AgentWorkspacePayloadService payloadService,
            AgentWorkspaceCutoverService cutoverService,
            AgentWorkspaceEntryMapper entryMapper,
            AgentWorkspaceMigrationMapper migrationMapper,
            AgentWorkspaceMigrationItemMapper itemMapper,
            @Qualifier("agentWorkspaceMigrationExecutor") TaskExecutor migrationExecutor) {
        this.configService = configService;
        this.payloadService = payloadService;
        this.cutoverService = cutoverService;
        this.entryMapper = entryMapper;
        this.migrationMapper = migrationMapper;
        this.itemMapper = itemMapper;
        this.migrationExecutor = migrationExecutor;
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceMigration", allEntries = true)
    public Long start(String backendType, Long storageConfigId, String localPath) {
        AgentWorkspaceLocation target = configService.validateTarget(
                backendType, storageConfigId, localPath);
        AgentWorkspaceConfig current = configService.lockCurrent();
        if (!AgentWorkspaceConfigService.MIGRATION_IDLE.equals(current.getMigrationStatus())) {
            throw new BusinessException("已有智能体工作空间迁移正在进行");
        }
        AgentWorkspaceLocation source = new AgentWorkspaceLocation(
                current.getBackendType(), current.getStorageConfigId(), current.getLocalPath());
        if (source.equals(target)) {
            throw new BusinessException("目标存储与当前存储相同");
        }
        long total = entryMapper.selectCount(null);
        AgentWorkspaceMigration migration = AgentWorkspaceMigration.builder()
                .sourceBackendType(source.backendType())
                .sourceStorageConfigId(source.storageConfigId())
                .sourceLocalPath(source.localPath())
                .targetBackendType(target.backendType())
                .targetStorageConfigId(target.storageConfigId())
                .targetLocalPath(target.localPath())
                .status("copying")
                .totalCount(total)
                .copiedCount(0L)
                .failedCount(0L)
                .startedAt(LocalDateTime.now())
                .build();
        migrationMapper.insert(migration);
        configService.markMigration(migration.getId(), "copying");
        Long migrationId = migration.getId();
        TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
            @Override
            public void afterCommit() {
                migrationExecutor.execute(() -> run(migrationId));
            }
        });
        return migrationId;
    }

    public AgentWorkspaceMigration get(Long id) {
        AgentWorkspaceMigration migration = migrationMapper.selectById(id);
        if (migration == null) {
            throw new BusinessException("工作空间迁移任务不存在");
        }
        return migration;
    }

    public AgentWorkspaceMigration latest() {
        return migrationMapper.selectOne(new LambdaQueryWrapper<AgentWorkspaceMigration>()
                .orderByDesc(AgentWorkspaceMigration::getId)
                .last("LIMIT 1"));
    }

    @CacheEvict(value = "agentWorkspaceMigration", allEntries = true)
    public void rollback(Long migrationId) {
        cutoverService.rollback(migrationId);
    }

    @CacheEvict(value = "agentWorkspaceMigration", allEntries = true)
    public void dismissFailure(Long migrationId) {
        AgentWorkspaceMigration migration = getUncached(migrationId);
        if (!"failed".equals(migration.getStatus())) {
            throw new BusinessException("只有失败的迁移可以解除锁定");
        }
        configService.clearFailedMigration(migrationId);
    }

    public void test(String backendType, Long storageConfigId, String localPath) {
        AgentWorkspaceLocation target = configService.validateTarget(
                backendType, storageConfigId, localPath);
        AgentWorkspaceStoredPayload stored = payloadService.write(
                "{\"content\":\"agent-workspace-connection-test\"}", target);
        try {
            payloadService.verify(stored);
        } finally {
            payloadService.delete(stored);
        }
    }

    void run(Long migrationId) {
        try {
            AgentWorkspaceMigration migration = getUncached(migrationId);
            AgentWorkspaceLocation target = new AgentWorkspaceLocation(
                    migration.getTargetBackendType(),
                    migration.getTargetStorageConfigId(),
                    migration.getTargetLocalPath());
            List<AgentWorkspaceEntry> entries = entryMapper.selectList(
                    new LambdaQueryWrapper<AgentWorkspaceEntry>().orderByAsc(AgentWorkspaceEntry::getId));
            long copied = 0;
            for (AgentWorkspaceEntry entry : entries) {
                String payload = payloadService.read(entry);
                AgentWorkspaceStoredPayload stored = payloadService.write(payload, target);
                payloadService.verify(stored);
                itemMapper.insert(toItem(migrationId, entry, stored));
                copied++;
                migration.setCopiedCount(copied);
                migrationMapper.updateById(migration);
            }
            migration.setStatus("verifying");
            migrationMapper.updateById(migration);
            verifyItems(migrationId);
            migration.setStatus("cutover");
            migrationMapper.updateById(migration);
            configService.markMigration(migrationId, "cutover");
            cutoverService.cutover(migrationId);
        } catch (Exception failure) {
            fail(migrationId, failure);
        }
    }

    private void verifyItems(Long migrationId) {
        for (AgentWorkspaceMigrationItem item : itemMapper.selectList(
                new LambdaQueryWrapper<AgentWorkspaceMigrationItem>()
                        .eq(AgentWorkspaceMigrationItem::getMigrationId, migrationId))) {
            payloadService.verify(targetPayload(item));
        }
    }

    private AgentWorkspaceMigrationItem toItem(
            Long migrationId,
            AgentWorkspaceEntry source,
            AgentWorkspaceStoredPayload target) {
        return AgentWorkspaceMigrationItem.builder()
                .migrationId(migrationId)
                .entryId(source.getId())
                .sourceBackendType(source.getBackendType())
                .sourceStorageConfigId(source.getStorageConfigId())
                .sourceLocalPath(source.getLocalPath())
                .sourceContentRef(source.getContentRef())
                .sourcePayload(source.getPayload())
                .targetBackendType(target.backendType())
                .targetStorageConfigId(target.storageConfigId())
                .targetLocalPath(target.localPath())
                .targetContentRef(target.contentRef())
                .targetPayload(target.databasePayload())
                .contentSha256(target.sha256())
                .contentSize(target.size())
                .status("copied")
                .build();
    }

    private AgentWorkspaceStoredPayload targetPayload(AgentWorkspaceMigrationItem item) {
        return new AgentWorkspaceStoredPayload(
                item.getTargetBackendType(),
                item.getTargetStorageConfigId(),
                item.getTargetLocalPath(),
                item.getTargetContentRef(),
                item.getTargetPayload(),
                item.getContentSha256(),
                item.getContentSize());
    }

    private void fail(Long migrationId, Exception failure) {
        for (AgentWorkspaceMigrationItem item : itemMapper.selectList(
                new LambdaQueryWrapper<AgentWorkspaceMigrationItem>()
                        .eq(AgentWorkspaceMigrationItem::getMigrationId, migrationId))) {
            payloadService.delete(targetPayload(item));
        }
        AgentWorkspaceMigration migration = migrationMapper.selectById(migrationId);
        if (migration != null) {
            migration.setStatus("failed");
            migration.setFailedCount(Math.max(1, migration.getFailedCount() == null
                    ? 1 : migration.getFailedCount() + 1));
            migration.setErrorMessage(abbreviate(failure.getMessage()));
            migration.setFinishedAt(LocalDateTime.now());
            migrationMapper.updateById(migration);
        }
        configService.markMigrationFailed(migrationId);
    }

    private AgentWorkspaceMigration getUncached(Long migrationId) {
        return Objects.requireNonNull(
                migrationMapper.selectById(migrationId), "migration disappeared");
    }

    private String abbreviate(String message) {
        if (message == null) {
            return "未知迁移错误";
        }
        return message.length() <= 2000 ? message : message.substring(0, 2000);
    }
}
