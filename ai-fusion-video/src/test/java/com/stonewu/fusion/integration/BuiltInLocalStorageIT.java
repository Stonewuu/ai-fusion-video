package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.media.ManagedStorageConfiguration;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.storage.BuiltInLocalStorage;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.channels.Channels;
import java.nio.file.Files;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 真实媒体入库验证：内置本地无需配置行，切换用户存储后历史对象仍从原后端读取。 */
@Testcontainers
class BuiltInLocalStorageIT extends MediaResourceTestContext {
    @Test
    void noUserConfigurationStoresAndReadsMediaWithoutCreatingAConfigurationRow() throws Exception {
        storageMapper.delete(new LambdaQueryWrapper<>());
        var historical = directory.resolve("historical.txt");
        Files.writeString(historical, "existing-file");
        var resource = upload(id(), png, "builtin.png", null, USER);
        var blob = repository.blob(repository.get(resource.id(), false).blobId());
        assertThat(blob.storageConfigId()).isEqualTo(BuiltInLocalStorage.CONFIG_ID);
        assertThat(blob.backend()).isEqualTo("local");
        assertThat(storageMapper.selectCount(new LambdaQueryWrapper<>())).isZero();
        try (var input = Channels.newInputStream(blobs.open(blob))) {
            assertThat(input.readAllBytes()).isEqualTo(png);
        }
        assertThat(Files.readString(historical)).isEqualTo("existing-file");
    }

    @Test
    void userStorageOverridesNewWritesWithoutMovingOrLosingBuiltInMedia() throws Exception {
        storageMapper.delete(new LambdaQueryWrapper<>());
        var previous = upload(id(), png, "before.png", null, USER);
        var oldBlob = repository.blob(repository.get(previous.id(), false).blobId());
        var custom = StorageConfig.builder().name("用户存储").type("local")
                .basePath(directory.resolve("custom").toString()).isDefault(true).status(1).build();
        storageMapper.insert(custom);
        var next = upload(id(), png, "after.png", null, USER);
        var newBlob = repository.blob(repository.get(next.id(), false).blobId());
        assertThat(newBlob.storageConfigId()).isEqualTo(custom.getId());
        assertThat(oldBlob.storageConfigId()).isEqualTo(BuiltInLocalStorage.CONFIG_ID);
        try (var input = Channels.newInputStream(blobs.open(oldBlob))) {
            assertThat(input.readAllBytes()).isEqualTo(png);
        }
        try (var input = Channels.newInputStream(blobs.open(newBlob))) {
            assertThat(input.readAllBytes()).isEqualTo(png);
        }
        assertThat(storageMapper.selectCount(new LambdaQueryWrapper<>())).isEqualTo(1);
    }

    @Test
    void deletingTheLastUserConfigurationRestoresBuiltInStorageImmediately() {
        var storage = context.getBean(ManagedStorageConfiguration.class);
        assertThat(storage.requireDefault().getId()).isEqualTo(71L);
        storageMapper.delete(new LambdaQueryWrapper<>());
        assertThat(storage.requireDefault().getId()).isEqualTo(BuiltInLocalStorage.CONFIG_ID);
        assertThat(storageMapper.selectCount(new LambdaQueryWrapper<>())).isZero();
    }

    @Test
    void anExistingDisabledUserConfigurationDoesNotSilentlySwitchStorage() {
        var custom = storageMapper.selectById(71L);
        custom.setStatus(0);
        storageMapper.updateById(custom);
        assertThatThrownBy(() -> context.getBean(ManagedStorageConfiguration.class).requireDefault())
                .isInstanceOfSatisfying(MediaException.class, failure ->
                        assertThat(failure.getErrorCode()).isEqualTo("MEDIA_STORAGE_CONFIGURATION_INVALID"));
    }
}
