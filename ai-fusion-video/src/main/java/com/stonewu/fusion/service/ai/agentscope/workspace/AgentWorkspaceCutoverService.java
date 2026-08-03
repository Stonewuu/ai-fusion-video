package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentWorkspaceConfig;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.entity.ai.AgentWorkspaceMigration;
import com.stonewu.fusion.entity.ai.AgentWorkspaceMigrationItem;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceConfigMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceMigrationItemMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceMigrationMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;

@Service
@RequiredArgsConstructor
public class AgentWorkspaceCutoverService {

    private final AgentWorkspaceConfigMapper configMapper;
    private final AgentWorkspaceEntryMapper entryMapper;
    private final AgentWorkspaceMigrationMapper migrationMapper;
    private final AgentWorkspaceMigrationItemMapper itemMapper;

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public void cutover(Long migrationId) {
        AgentWorkspaceMigration migration = requireMigration(migrationId);
        AgentWorkspaceConfig config = requireLockedConfig(migrationId);
        List<AgentWorkspaceMigrationItem> items = items(migrationId);
        for (AgentWorkspaceMigrationItem item : items) {
            AgentWorkspaceEntry entry = entryMapper.selectById(item.getEntryId());
            if (entry == null || !item.getContentSha256().equals(entry.getContentSha256())) {
                throw new BusinessException("迁移切换前工作空间条目发生变化: " + item.getEntryId());
            }
            applyTarget(entry, item);
            entryMapper.updateById(entry);
        }
        config.setBackendType(migration.getTargetBackendType());
        config.setStorageConfigId(migration.getTargetStorageConfigId());
        config.setLocalPath(migration.getTargetLocalPath());
        config.setMigrationStatus(AgentWorkspaceConfigService.MIGRATION_IDLE);
        config.setActiveMigrationId(null);
        configMapper.updateById(config);
        migration.setStatus("completed");
        migration.setFinishedAt(LocalDateTime.now());
        migrationMapper.updateById(migration);
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public void rollback(Long migrationId) {
        AgentWorkspaceMigration migration = requireMigration(migrationId);
        if (!"completed".equals(migration.getStatus())) {
            throw new BusinessException("只有已完成的工作空间迁移可以回滚");
        }
        AgentWorkspaceConfig config = configMapper.selectCurrentForUpdate();
        if (config == null || !sameLocation(config, migration)) {
            throw new BusinessException("当前存储已经变化，不能回滚此迁移");
        }
        for (AgentWorkspaceMigrationItem item : items(migrationId)) {
            AgentWorkspaceEntry entry = entryMapper.selectById(item.getEntryId());
            if (entry == null) {
                continue;
            }
            applySource(entry, item);
            entryMapper.updateById(entry);
        }
        config.setBackendType(migration.getSourceBackendType());
        config.setStorageConfigId(migration.getSourceStorageConfigId());
        config.setLocalPath(migration.getSourceLocalPath());
        config.setMigrationStatus(AgentWorkspaceConfigService.MIGRATION_IDLE);
        config.setActiveMigrationId(null);
        configMapper.updateById(config);
        migration.setStatus("rolled_back");
        migration.setFinishedAt(LocalDateTime.now());
        migrationMapper.updateById(migration);
    }

    private AgentWorkspaceConfig requireLockedConfig(Long migrationId) {
        AgentWorkspaceConfig config = configMapper.selectCurrentForUpdate();
        if (config == null || !migrationId.equals(config.getActiveMigrationId())) {
            throw new BusinessException("工作空间迁移任务已变化，不能执行切换");
        }
        return config;
    }

    private AgentWorkspaceMigration requireMigration(Long migrationId) {
        AgentWorkspaceMigration migration = migrationMapper.selectById(migrationId);
        if (migration == null) {
            throw new BusinessException("工作空间迁移任务不存在");
        }
        return migration;
    }

    private List<AgentWorkspaceMigrationItem> items(Long migrationId) {
        return itemMapper.selectList(new LambdaQueryWrapper<AgentWorkspaceMigrationItem>()
                .eq(AgentWorkspaceMigrationItem::getMigrationId, migrationId)
                .orderByAsc(AgentWorkspaceMigrationItem::getId));
    }

    private boolean sameLocation(AgentWorkspaceConfig config, AgentWorkspaceMigration migration) {
        return migration.getTargetBackendType().equals(config.getBackendType())
                && Objects.equals(
                        migration.getTargetStorageConfigId(), config.getStorageConfigId())
                && Objects.equals(migration.getTargetLocalPath(), config.getLocalPath());
    }

    private void applyTarget(AgentWorkspaceEntry entry, AgentWorkspaceMigrationItem item) {
        entry.setBackendType(item.getTargetBackendType());
        entry.setStorageConfigId(item.getTargetStorageConfigId());
        entry.setLocalPath(item.getTargetLocalPath());
        entry.setContentRef(item.getTargetContentRef());
        entry.setPayload(item.getTargetPayload());
    }

    private void applySource(AgentWorkspaceEntry entry, AgentWorkspaceMigrationItem item) {
        entry.setBackendType(item.getSourceBackendType());
        entry.setStorageConfigId(item.getSourceStorageConfigId());
        entry.setLocalPath(item.getSourceLocalPath());
        entry.setContentRef(item.getSourceContentRef());
        entry.setPayload(item.getSourcePayload());
    }
}
