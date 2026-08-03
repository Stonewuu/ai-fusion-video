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
import com.stonewu.fusion.service.storage.StorageConfigReferenceGuard;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class AgentWorkspaceStorageReferenceGuard implements StorageConfigReferenceGuard {

    private final AgentWorkspaceConfigMapper configMapper;
    private final AgentWorkspaceEntryMapper entryMapper;
    private final AgentWorkspaceMigrationMapper migrationMapper;
    private final AgentWorkspaceMigrationItemMapper migrationItemMapper;

    @Override
    public void assertDeletable(Long storageConfigId) {
        if (storageConfigId == null) {
            return;
        }
        boolean referenced = configMapper.selectCount(new LambdaQueryWrapper<AgentWorkspaceConfig>()
                        .eq(AgentWorkspaceConfig::getStorageConfigId, storageConfigId)) > 0
                || entryMapper.selectCount(new LambdaQueryWrapper<AgentWorkspaceEntry>()
                        .eq(AgentWorkspaceEntry::getStorageConfigId, storageConfigId)) > 0
                || migrationMapper.selectCount(new LambdaQueryWrapper<AgentWorkspaceMigration>()
                        .and(wrapper -> wrapper
                                .eq(AgentWorkspaceMigration::getSourceStorageConfigId, storageConfigId)
                                .or()
                                .eq(AgentWorkspaceMigration::getTargetStorageConfigId, storageConfigId))) > 0
                || migrationItemMapper.selectCount(
                        new LambdaQueryWrapper<AgentWorkspaceMigrationItem>()
                                .and(wrapper -> wrapper
                                        .eq(AgentWorkspaceMigrationItem::getSourceStorageConfigId,
                                                storageConfigId)
                                        .or()
                                        .eq(AgentWorkspaceMigrationItem::getTargetStorageConfigId,
                                                storageConfigId))) > 0;
        if (referenced) {
            throw new BusinessException("该存储配置仍被智能体工作空间或迁移历史引用，不能删除");
        }
    }
}
