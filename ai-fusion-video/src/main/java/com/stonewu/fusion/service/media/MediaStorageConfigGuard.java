package com.stonewu.fusion.service.media;

import com.stonewu.fusion.mapper.media.MediaBlobMapper;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageConfigReferenceGuard;
import com.stonewu.fusion.service.storage.StorageConfigUpdateGuard;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

import java.util.Objects;

/**
 * 允许凭据轮换和开关状态，不允许原地迁移已有对象的地址。配置行锁与登记事务共用。
 */
@Component
@RequiredArgsConstructor
public class MediaStorageConfigGuard implements StorageConfigReferenceGuard, StorageConfigUpdateGuard {
    private final MediaBlobMapper blobs;

    @Override
    public void assertDeletable(Long id) {
        if (referenced(id)) {
            throw new BusinessException(409, "存储配置仍被托管媒体引用，不能删除");
        }
    }

    @Override
    public void assertUpdatable(StorageConfig old, StorageConfig patch) {
        if (!referenced(old.getId())) {
            return;
        }
        if (changed(old.getType(), patch.getType()) || changed(old.getProvider(), patch.getProvider())
                || changed(old.getBasePath(), patch.getBasePath()) || changed(old.getBucketName(),
                patch.getBucketName())
                || changed(old.getEndpoint(), patch.getEndpoint()) || changed(old.getRegion(), patch.getRegion())) {
            throw new BusinessException(409, "此存储仍有托管媒体，不能直接修改其后端、目录、桶或端点");
        }
    }

    private boolean referenced(Long id) {
        return blobs.countStorageReferences(id) > 0;
    }

    private boolean changed(String old, String next) {
        return next != null && !Objects.equals(old, next);
    }
}
