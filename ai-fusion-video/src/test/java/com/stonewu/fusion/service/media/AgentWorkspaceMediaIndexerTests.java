package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspacePayloadService;
import com.stonewu.fusion.service.media.lifecycle.AgentWorkspaceMediaIndexer;
import org.junit.jupiter.api.Test;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import java.util.List;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class AgentWorkspaceMediaIndexerTests {
    private final AgentWorkspaceEntryMapper entries = mock(AgentWorkspaceEntryMapper.class);
    private final AgentWorkspacePayloadService payloads = mock(AgentWorkspacePayloadService.class);
    private final AgentWorkspaceMediaIndexer indexer = new AgentWorkspaceMediaIndexer(entries, payloads,
            new ObjectMapper());

    @Test
    void indexesVerifiedExternalContentWithVersionAndDigestGuards() {
        String body = "{\"content\":\"media://d20c41fc-9f42-4c84-bda4-b20f29472da0\"}";
        var entry = entry(body);
        when(payloads.read(entry)).thenReturn(body);
        when(entries.selectCount(any())).thenReturn(0L);
        assertThat(indexer.indexBatch()).isTrue();
        verify(entries).indexMediaReferences(eq("[\"media://d20c41fc-9f42-4c84-bda4-b20f29472da0\"]"), eq(8L), eq(4L),
                eq(entry.getContentSha256()));
    }

    @Test
    void unavailableOrCorruptExternalContentLeavesCleanupBlocked() {
        var entry = entry("{}");
        when(payloads.read(entry)).thenReturn("{\"different\":true}");
        when(entries.selectCount(any())).thenReturn(1L);
        assertThat(indexer.indexBatch()).isFalse();
        verify(entries, never()).indexMediaReferences(anyString(), anyLong(), anyLong(), anyString());
    }

    private AgentWorkspaceEntry entry(String body) {
        var entry = AgentWorkspaceEntry.builder().id(8L).version(4L).backendType("local")
                .contentSha256(AgentWorkspacePayloadService.sha256(body)).build();
        when(entries.selectPage(any(Page.class), any())).thenReturn(
                new Page<AgentWorkspaceEntry>().setRecords(List.of(entry)));
        when(entries.selectById(8L)).thenReturn(entry);
        return entry;
    }
}
