package com.stonewu.fusion.service.storage;

@FunctionalInterface
public interface StorageConfigReferenceGuard {

    void assertDeletable(Long storageConfigId);
}
