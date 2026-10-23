package com.stonewu.fusion.service.media.lifecycle;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspacePayloadService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.stereotype.Component;

/**
 * 老工作空间可能在本地或对象存储中；读取完整正文后以版本和摘要条件提交索引。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AgentWorkspaceMediaIndexer {
    private final AgentWorkspaceEntryMapper entries;
    private final AgentWorkspacePayloadService payloads;
    private final ObjectMapper mapper;


    public boolean indexBatch() {
        var pending = entries.selectPage(new Page<AgentWorkspaceEntry>(1, 20, false),
                        Wrappers.<AgentWorkspaceEntry>lambdaQuery()
                                .select(AgentWorkspaceEntry::getId).isNull(AgentWorkspaceEntry::getMediaReferencesJson)
                                .orderByAsc(AgentWorkspaceEntry::getId)).getRecords().stream().map(AgentWorkspaceEntry::getId)
                .toList();
        for (long id : pending) {
            try {
                var entry = entries.selectById(id);
                if (entry == null || entry.getMediaReferencesJson() != null) {
                    continue;
                }
                String payload = payloads.read(entry);
                if (!AgentWorkspacePayloadService.sha256(payload).equals(entry.getContentSha256())) {
                    throw new IllegalStateException("工作空间正文摘要不匹配");
                }
                String references = MediaReferenceIndex.extract(payload, mapper);
                entries.indexMediaReferences(references, id, entry.getVersion(), entry.getContentSha256());
            } catch (RuntimeException failure) {
                // 缺失或变更中的历史正文不能当成无引用；索引完成前停止物理清理。
                log.warn("工作空间媒体索引未完成 [entryId={}, failureType={}]", id, failure.getClass().getSimpleName());
            }
        }
        return entries.selectCount(
                Wrappers.<AgentWorkspaceEntry>lambdaQuery().isNull(AgentWorkspaceEntry::getMediaReferencesJson)) == 0;
    }
}
