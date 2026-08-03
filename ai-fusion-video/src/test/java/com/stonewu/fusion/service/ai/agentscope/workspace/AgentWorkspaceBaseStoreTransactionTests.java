package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentWorkspaceBaseStoreTransactionTests {

    private final AgentWorkspaceEntryMapper entryMapper = mock(AgentWorkspaceEntryMapper.class);
    private final AgentWorkspaceConfigService configService = mock(AgentWorkspaceConfigService.class);
    private final AgentWorkspacePayloadService payloadService = mock(AgentWorkspacePayloadService.class);
    private final AgentWorkspaceBaseStore store = new AgentWorkspaceBaseStore(
            entryMapper, configService, payloadService, new ObjectMapper());

    @AfterEach
    void clearTransactionSynchronization() {
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.clearSynchronization();
        }
        TransactionSynchronizationManager.setActualTransactionActive(false);
    }

    @Test
    void updateDeletesPreviousPayloadOnlyAfterCommit() {
        Fixture fixture = fixture();
        beginTransaction();

        store.put(List.of("namespace"), "/skill/SKILL.md", Map.of("content", "new"));

        verify(payloadService, never()).delete(fixture.previous());
        verify(payloadService, never()).delete(fixture.next());
        TransactionSynchronizationManager.getSynchronizations()
                .forEach(TransactionSynchronization::afterCommit);
        TransactionSynchronizationManager.getSynchronizations()
                .forEach(sync -> sync.afterCompletion(TransactionSynchronization.STATUS_COMMITTED));

        verify(payloadService).delete(fixture.previous());
        verify(payloadService, never()).delete(fixture.next());
    }

    @Test
    void updateKeepsPreviousPayloadAndDeletesNewPayloadAfterRollback() {
        Fixture fixture = fixture();
        beginTransaction();

        store.put(List.of("namespace"), "/skill/SKILL.md", Map.of("content", "new"));

        TransactionSynchronizationManager.getSynchronizations()
                .forEach(sync -> sync.afterCompletion(TransactionSynchronization.STATUS_ROLLED_BACK));

        verify(payloadService, never()).delete(fixture.previous());
        verify(payloadService).delete(fixture.next());
    }

    private Fixture fixture() {
        String namespaceHash = AgentWorkspacePayloadService.sha256("namespace");
        String itemHash = AgentWorkspacePayloadService.sha256("/skill/SKILL.md");
        AgentWorkspaceEntry current = AgentWorkspaceEntry.builder()
                .id(7L)
                .namespaceHash(namespaceHash)
                .namespaceKey("namespace")
                .itemHash(itemHash)
                .itemKey("/skill/SKILL.md")
                .backendType(AgentWorkspaceBackend.LOCAL)
                .localPath("D:/workspace")
                .contentRef("objects/previous.json")
                .contentSha256("previous")
                .contentSize(8L)
                .version(1L)
                .build();
        AgentWorkspaceLocation location = new AgentWorkspaceLocation(
                AgentWorkspaceBackend.LOCAL, null, "D:/workspace");
        AgentWorkspaceStoredPayload previous = new AgentWorkspaceStoredPayload(
                AgentWorkspaceBackend.LOCAL,
                null,
                "D:/workspace",
                "objects/previous.json",
                null,
                "previous",
                8L);
        AgentWorkspaceStoredPayload next = new AgentWorkspaceStoredPayload(
                AgentWorkspaceBackend.LOCAL,
                null,
                "D:/workspace",
                "objects/next.json",
                null,
                "next",
                7L);
        when(entryMapper.selectForUpdate(namespaceHash, itemHash)).thenReturn(current);
        when(configService.lockWritableLocation()).thenReturn(location);
        when(payloadService.write(anyString(), eq(location))).thenReturn(next);
        return new Fixture(previous, next);
    }

    private void beginTransaction() {
        TransactionSynchronizationManager.setActualTransactionActive(true);
        TransactionSynchronizationManager.initSynchronization();
    }

    private record Fixture(
            AgentWorkspaceStoredPayload previous,
            AgentWorkspaceStoredPayload next) {
    }
}
