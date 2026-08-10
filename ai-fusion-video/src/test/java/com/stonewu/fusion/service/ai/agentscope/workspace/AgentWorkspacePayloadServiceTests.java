package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import com.stonewu.fusion.service.storage.StorageConfigService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class AgentWorkspacePayloadServiceTests {

    @TempDir
    Path temporaryDirectory;

    @Test
    void databasePayloadRoundTripsWithoutExternalStorage() {
        AgentWorkspacePayloadService service = service();

        AgentWorkspaceStoredPayload stored = service.write(
                "{\"content\":\"hello\"}",
                new AgentWorkspaceLocation(AgentWorkspaceBackend.DATABASE, null, null));

        assertThat(stored.databasePayload()).isEqualTo("{\"content\":\"hello\"}");
        assertThat(stored.contentRef()).isNull();
        assertThat(service.read(stored)).isEqualTo("{\"content\":\"hello\"}");
        service.verify(stored);
    }

    @Test
    void localPayloadUsesConfiguredPersistentRootAndCanBeCleanedUp() {
        AgentWorkspacePayloadService service = service();

        AgentWorkspaceStoredPayload stored = service.write(
                "{\"content\":\"local\"}",
                new AgentWorkspaceLocation(
                        AgentWorkspaceBackend.LOCAL,
                        null,
                        temporaryDirectory.toString()));

        Path storedFile = temporaryDirectory.resolve(stored.contentRef()).normalize();
        assertThat(stored.localPath()).isEqualTo(temporaryDirectory.toAbsolutePath().toString());
        assertThat(storedFile).exists();
        assertThat(service.read(stored)).isEqualTo("{\"content\":\"local\"}");
        service.verify(stored);

        service.delete(stored);
        assertThat(Files.exists(storedFile)).isFalse();
    }

    private AgentWorkspacePayloadService service() {
        return new AgentWorkspacePayloadService(
                mock(StorageConfigService.class),
                mock(S3StorageConfigResolver.class),
                mock(S3ClientFactory.class),
                temporaryDirectory.toString());
    }
}
