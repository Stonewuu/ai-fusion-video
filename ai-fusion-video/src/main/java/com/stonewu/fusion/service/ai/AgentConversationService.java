package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStateCleanupPolicyService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.util.List;
import java.util.function.Supplier;

/**
 * Agent 对话索引服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentConversationService {

    private final AgentConversationMapper conversationMapper;
    private final AgentStatePreflight statePreflight;
    private final AgentStateCleanupPolicyService stateCleanupPolicy;
    private final AgentRunMapper runMapper;
    private final AgentRuntimeSchedulers schedulers;

    @Transactional
    public AgentConversation createOrUpdate(String conversationId, Long userId, Long projectId,
                                            String contextType, Long contextId,
                                            String agentType, String title, String category) {
        AgentConversation existing =
                conversationMapper.selectByConversationIdForUpdate(conversationId);
        if (existing != null) {
            LocalDateTime updatedAt = conversationMapper.selectDatabaseNow();
            stateCleanupPolicy.requireAvailable(
                    existing,
                    updatedAt,
                    stateCleanupPolicy.getCurrent().getRetentionDays());
            boolean assistantConversation = "assistant".equals(existing.getCategory())
                    || "assistant".equals(category);
            String existingTitle = existing.getTitle() == null
                    ? null
                    : existing.getTitle().trim();
            boolean placeholderTitle = existingTitle == null
                    || existingTitle.isBlank()
                    || "新对话".equals(existingTitle);
            boolean updateTitle = title != null && (!assistantConversation || placeholderTitle);
            LambdaUpdateWrapper<AgentConversation> metadataUpdate =
                    new LambdaUpdateWrapper<AgentConversation>()
                            .eq(AgentConversation::getConversationId, conversationId)
                            .eq(AgentConversation::getDeleted, false)
                            .set(updateTitle, AgentConversation::getTitle, title)
                            .set(existing.getCategory() == null && category != null,
                                    AgentConversation::getCategory, category)
                            .set(AgentConversation::getStatus, "running")
                            .set(AgentConversation::getUpdateTime, updatedAt);
            if (conversationMapper.update(null, metadataUpdate) != 1) {
                throw new IllegalStateException("Agent conversation metadata update failed: "
                        + conversationId);
            }
            if (updateTitle) {
                existing.setTitle(title);
            }
            if (existing.getCategory() == null && category != null) {
                existing.setCategory(category);
            }
            existing.setStatus("running");
            existing.setUpdateTime(updatedAt);
            return existing;
        }

        AgentConversation conv = AgentConversation.builder()
                .conversationId(conversationId)
                .userId(userId)
                .projectId(projectId)
                .contextType(contextType)
                .contextId(contextId)
                .agentType(agentType)
                .category(category)
                .title(title != null ? title : "新对话")
                .status("running")
                .messageCount(0)
                .nextMessageOrder(1L)
                .agentStateLastActiveAt(conversationMapper.selectDatabaseNow())
                .build();
        conversationMapper.insert(conv);
        return conv;
    }

    @Transactional
    public void finish(String conversationId, String status) {
        LocalDateTime finishedAt = conversationMapper.selectDatabaseNow();
        conversationMapper.update(null, new LambdaUpdateWrapper<AgentConversation>()
                .eq(AgentConversation::getConversationId, conversationId)
                .eq(AgentConversation::getDeleted, false)
                .set(AgentConversation::getStatus, status)
                .set(AgentConversation::getAgentStateLastActiveAt, finishedAt)
                .set(AgentConversation::getAgentStateExpiredAt, null)
                .set(AgentConversation::getUpdateTime, finishedAt));
    }

    public AgentConversation getByConversationId(String conversationId) {
        return conversationMapper.selectOne(
                new LambdaQueryWrapper<AgentConversation>()
                        .eq(AgentConversation::getConversationId, conversationId)
                        .eq(AgentConversation::getDeleted, false));
    }

    /**
     * Resolves a conversation only when it belongs to the supplied user.
     *
     * Conversation ids are exposed to the browser, so callers that return
     * message/history data must never use the unscoped lookup above.
     */
    public AgentConversation getOwnedByConversationId(String conversationId, long userId) {
        return conversationMapper.selectOne(new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getConversationId, conversationId)
                .eq(AgentConversation::getUserId, userId)
                .eq(AgentConversation::getDeleted, false));
    }

    public PageResult<AgentConversation> listByUser(Long userId, int pageNo, int pageSize) {
        return annotateStateStatus(PageResult.of(conversationMapper.selectPage(
                new Page<>(pageNo, pageSize),
                new LambdaQueryWrapper<AgentConversation>()
                        .eq(AgentConversation::getUserId, userId)
                        .eq(AgentConversation::getDeleted, false)
                        .orderByDesc(AgentConversation::getUpdateTime)
                        .orderByDesc(AgentConversation::getId))));
    }

    public PageResult<AgentConversation> listByUserAndCategory(Long userId, String category, int pageNo, int pageSize) {
        return annotateStateStatus(PageResult.of(conversationMapper.selectPage(
                new Page<>(pageNo, pageSize),
                new LambdaQueryWrapper<AgentConversation>()
                        .eq(AgentConversation::getUserId, userId)
                        .eq(AgentConversation::getCategory, category)
                        .eq(AgentConversation::getDeleted, false)
                        .orderByDesc(AgentConversation::getUpdateTime)
                        .orderByDesc(AgentConversation::getId))));
    }

    public List<AgentConversation> listByProject(Long projectId) {
        return annotateStateStatus(conversationMapper.selectList(
                new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getProjectId, projectId)
                .eq(AgentConversation::getDeleted, false)
                        .orderByDesc(AgentConversation::getUpdateTime)));
    }

    /**
     * 查询用户正在运行的 pipeline 对话（status=running）
     */
    public List<AgentConversation> listRunning(Long userId) {
        return conversationMapper.selectList(new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getUserId, userId)
                .eq(AgentConversation::getStatus, "running")
                .eq(AgentConversation::getDeleted, false)
                .orderByDesc(AgentConversation::getUpdateTime));
    }

    public Mono<Void> deleteConversation(long id, long currentUserId) {
        return deleteOwnedConversation(
                () -> ownedConversation(id, currentUserId), currentUserId);
    }

    public Mono<Void> deleteConversationByConversationId(
            String conversationId,
            long currentUserId) {
        return deleteOwnedConversation(
                () -> ownedConversation(conversationId, currentUserId), currentUserId);
    }

    private Mono<Void> deleteOwnedConversation(
            Supplier<LambdaQueryWrapper<AgentConversation>> ownedConversation,
            long currentUserId) {
        return Mono.fromCallable(() -> conversationMapper.selectOne(ownedConversation.get()))
                .subscribeOn(schedulers.journal())
                .flatMap(conversation -> Mono.fromCallable(() ->
                                runMapper.selectStateSessionIdsByConversation(
                                        conversation.getConversationId()))
                        .subscribeOn(schedulers.journal())
                        .flatMapMany(Flux::fromIterable)
                        .concatMap(sessionId -> statePreflight.deleteSession(
                                String.valueOf(currentUserId), sessionId))
                        .then(Mono.fromRunnable(() -> conversationMapper.delete(
                                        ownedConversation.get()))
                                .subscribeOn(schedulers.journal())))
                .then();
    }

    private LambdaQueryWrapper<AgentConversation> ownedConversation(long id, long userId) {
        return new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getId, id)
                .eq(AgentConversation::getUserId, userId)
                .eq(AgentConversation::getDeleted, false);
    }

    private LambdaQueryWrapper<AgentConversation> ownedConversation(
            String conversationId,
            long userId) {
        return new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getConversationId, conversationId)
                .eq(AgentConversation::getUserId, userId)
                .eq(AgentConversation::getDeleted, false);
    }

    private PageResult<AgentConversation> annotateStateStatus(
            PageResult<AgentConversation> page) {
        annotateStateStatus(page.getList());
        return page;
    }

    private List<AgentConversation> annotateStateStatus(
            List<AgentConversation> conversations) {
        if (conversations.isEmpty()) {
            return conversations;
        }
        LocalDateTime databaseNow = conversationMapper.selectDatabaseNow();
        int retentionDays = stateCleanupPolicy.getCurrent().getRetentionDays();
        conversations.forEach(conversation -> conversation.setAgentStateStatus(
                stateCleanupPolicy.stateStatus(
                        conversation, databaseNow, retentionDays)));
        return conversations;
    }
}
