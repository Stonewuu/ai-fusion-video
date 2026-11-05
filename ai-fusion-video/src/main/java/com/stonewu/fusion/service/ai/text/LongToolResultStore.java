package com.stonewu.fusion.service.ai.text;

import com.stonewu.fusion.service.resource.ScriptFields;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 工具结果只在产生结果的用户会话内可读；模型只能提交结果编号，不能指定命名空间或文件路径。
 */
@Component
public final class LongToolResultStore {
    private final AgentWorkspaceBaseStore store;

    public LongToolResultStore(AgentWorkspaceBaseStore store) {
        this.store = Objects.requireNonNull(store);
    }

    public String save(String userId, String sessionId, String content) {
        var namespace = namespace(userId, sessionId);
        String id = TextPages.version(content);
        StoreItem existing = store.get(namespace, id);
        if (existing == null && !store.putArtifactIfAbsent(namespace, id, Map.of("content", content))) {
            existing = store.get(namespace, id);
            if (existing == null) {
                throw new IllegalStateException("长工具结果未能保存，请重试任务");
            }
        }
        if (existing != null && !content.equals(existing.value().get("content"))) {
            throw new IllegalStateException("已保存的长工具结果与内容版本不一致");
        }
        return id;
    }

    public TextPages.Page read(String userId, String sessionId, String resultId, TextPages.Request request) {
        if (resultId == null || !resultId.matches("[0-9a-f]{64}")) {
            throw ScriptFields.invalid("resultId 必须使用工具返回的结果编号");
        }
        StoreItem item = store.get(namespace(userId, sessionId), resultId);
        if (item == null) {
            throw ScriptFields.invalid("当前会话中没有此工具结果，请重新执行原查询");
        }
        Object value = item.value().get("content");
        if (!(value instanceof String content) || !resultId.equals(TextPages.version(content))) {
            throw new IllegalStateException("已保存的长工具结果校验失败");
        }
        return TextPages.read(content, request);
    }

    private List<String> namespace(String userId, String sessionId) {
        if (TransactionSynchronizationManager.isActualTransactionActive()) {
            throw new IllegalStateException("工具结果文件读写不能在数据库事务内执行");
        }
        if (userId == null || userId.isBlank() || sessionId == null || sessionId.isBlank()) {
            throw new IllegalStateException("读取工具结果需要可信的用户和会话上下文");
        }
        // 会话编号先摘要，避免分隔符使不同会话得到相同的工作空间命名空间。
        return List.of("tool-results", userId, TextPages.version(sessionId));
    }
}
