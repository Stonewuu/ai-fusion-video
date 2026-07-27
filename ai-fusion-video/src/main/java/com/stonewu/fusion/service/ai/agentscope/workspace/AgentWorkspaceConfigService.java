package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentWorkspaceConfig;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceConfigMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.storage.StorageTypes;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.file.Files;
import java.nio.file.Path;

@Service
@RequiredArgsConstructor
public class AgentWorkspaceConfigService {

    public static final long CONFIG_ID = 1L;
    public static final String MIGRATION_IDLE = "idle";

    private final AgentWorkspaceConfigMapper configMapper;
    private final AgentWorkspaceEntryMapper entryMapper;
    private final StorageConfigService storageConfigService;
    private final AgentWorkspacePayloadService payloadService;

    @Cacheable(value = "agentWorkspaceConfig", key = "'current'")
    public AgentWorkspaceConfig getCurrent() {
        AgentWorkspaceConfig config = configMapper.selectById(CONFIG_ID);
        if (config == null) {
            throw new BusinessException("智能体工作空间配置不存在，请先执行数据库迁移");
        }
        return config;
    }

    public AgentWorkspaceLocation currentLocation() {
        AgentWorkspaceConfig config = getCurrent();
        return new AgentWorkspaceLocation(
                config.getBackendType(), config.getStorageConfigId(), config.getLocalPath());
    }

    @Transactional
    public AgentWorkspaceLocation lockWritableLocation() {
        AgentWorkspaceConfig config = configMapper.selectCurrentForUpdate();
        if (config == null) {
            throw new BusinessException("智能体工作空间配置不存在");
        }
        if (!MIGRATION_IDLE.equals(config.getMigrationStatus())) {
            throw new BusinessException("智能体工作空间正在迁移，暂时不能写入");
        }
        return new AgentWorkspaceLocation(
                config.getBackendType(), config.getStorageConfigId(), config.getLocalPath());
    }

    public AgentWorkspaceLocation validateTarget(
            String backendType,
            Long storageConfigId,
            String localPath) {
        String type = AgentWorkspaceBackend.requireSupported(backendType);
        if (AgentWorkspaceBackend.OBJECT_STORAGE.equals(type)) {
            if (storageConfigId == null) {
                throw new BusinessException("对象存储模式必须选择存储配置");
            }
            StorageConfig storage = storageConfigService.getById(storageConfigId);
            if (!Integer.valueOf(1).equals(storage.getStatus())
                    || !StorageTypes.isS3Like(storage.getType())) {
                throw new BusinessException("请选择已启用的 S3 兼容存储配置");
            }
            return new AgentWorkspaceLocation(type, storageConfigId, null);
        }
        if (AgentWorkspaceBackend.LOCAL.equals(type)) {
            Path root = payloadService.resolveLocalRoot(localPath);
            try {
                Files.createDirectories(root);
                if (!Files.isDirectory(root) || !Files.isWritable(root)) {
                    throw new BusinessException("本地工作空间目录不可写: " + root);
                }
            } catch (BusinessException failure) {
                throw failure;
            } catch (Exception failure) {
                throw new BusinessException("无法使用本地工作空间目录: " + failure.getMessage());
            }
            return new AgentWorkspaceLocation(type, null, root.toString());
        }
        return new AgentWorkspaceLocation(type, null, null);
    }

    public WorkspaceUsage usage() {
        return new WorkspaceUsage(entryMapper.selectCount(null), entryMapper.sumContentSize());
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public AgentWorkspaceConfig lockCurrent() {
        AgentWorkspaceConfig config = configMapper.selectCurrentForUpdate();
        if (config == null) {
            throw new BusinessException("智能体工作空间配置不存在");
        }
        return config;
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public void markMigration(Long migrationId, String status) {
        AgentWorkspaceConfig config = lockCurrent();
        config.setActiveMigrationId(migrationId);
        config.setMigrationStatus(status);
        configMapper.updateById(config);
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public void activate(AgentWorkspaceLocation location, Long migrationId) {
        AgentWorkspaceConfig config = lockCurrent();
        if (migrationId != null && !migrationId.equals(config.getActiveMigrationId())) {
            throw new BusinessException("工作空间迁移任务已变化，不能切换存储");
        }
        config.setBackendType(location.backendType());
        config.setStorageConfigId(location.storageConfigId());
        config.setLocalPath(location.localPath());
        config.setMigrationStatus(MIGRATION_IDLE);
        config.setActiveMigrationId(null);
        configMapper.updateById(config);
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public void markMigrationFailed(Long migrationId) {
        AgentWorkspaceConfig config = lockCurrent();
        if (migrationId.equals(config.getActiveMigrationId())) {
            config.setMigrationStatus("failed");
            configMapper.updateById(config);
        }
    }

    @Transactional
    @CacheEvict(value = "agentWorkspaceConfig", allEntries = true)
    public void clearFailedMigration(Long migrationId) {
        AgentWorkspaceConfig config = lockCurrent();
        if (migrationId.equals(config.getActiveMigrationId())) {
            config.setMigrationStatus(MIGRATION_IDLE);
            config.setActiveMigrationId(null);
            configMapper.updateById(config);
        }
    }

    public record WorkspaceUsage(long entryCount, long contentBytes) {
    }
}
