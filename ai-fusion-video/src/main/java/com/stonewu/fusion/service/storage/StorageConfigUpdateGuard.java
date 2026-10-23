package com.stonewu.fusion.service.storage;

import com.stonewu.fusion.entity.storage.StorageConfig;

@FunctionalInterface
public interface StorageConfigUpdateGuard {
    void assertUpdatable(StorageConfig current, StorageConfig patch);
}
