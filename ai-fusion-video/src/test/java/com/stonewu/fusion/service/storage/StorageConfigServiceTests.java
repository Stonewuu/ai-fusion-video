package com.stonewu.fusion.service.storage;

import org.junit.jupiter.api.Test;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.service.media.ManagedStorageConfiguration;
import org.springframework.beans.factory.support.StaticListableBeanFactory;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

class StorageConfigServiceTests {

    @Test
    void builtInConfigurationIsReadOnlyAndNeverTouchesTheConfigurationTable() {
        var configurations = mock(StorageConfigMapper.class);
        var beans = new StaticListableBeanFactory();
        var service = new StorageConfigService(configurations, mock(S3ClientFactory.class),
                beans.getBeanProvider(StorageConfigReferenceGuard.class),
                beans.getBeanProvider(StorageConfigUpdateGuard.class), new BuiltInLocalStorage("./data/media"),
                mock(ManagedStorageConfiguration.class));
        assertThat(service.getById(0L).getBasePath()).isEqualTo("./data/media");
        var builtIn = StorageConfig.builder().id(0L).type("local").build();
        for (Runnable mutation : List.<Runnable>of(() -> service.create(builtIn), () -> service.update(builtIn),
                () -> service.delete(0L), () -> service.setDefault(0L))) {
            assertThatThrownBy(mutation::run).isInstanceOf(BusinessException.class).hasMessageContaining("只读");
        }
        verifyNoInteractions(configurations);
    }
}
