package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Agent 对话索引服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentConversationService {

    private final AgentConversationMapper conversationMapper;
    private final AgentStatePreflight statePreflight;
    private final AgentRuntimeSchedulers schedulers;

    @Transactional
    public AgentConversation createOrUpdate(String conversationId, Long userId, Long projectId,
                                            String contextType, Long contextId,
                                            String agentType, String title, String category) {
        AgentConversation existing = conversationMapper.selectOne(
                new LambdaQueryWrapper<AgentConversation>().eq(AgentConversation::getConversationId, conversationId));
        if (existing != null) {
            if (title != null) existing.setTitle(title);
            existing.setLastMessageTime(LocalDateTime.now());
            existing.setMessageCount(existing.getMessageCount() + 1);
            conversationMapper.updateById(existing);
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
                .messageCount(1)
                .lastMessageTime(LocalDateTime.now())
                .build();
        conversationMapper.insert(conv);
        return conv;
    }

    @Transactional
    public void finish(String conversationId, String status) {
        AgentConversation conv = conversationMapper.selectOne(
                new LambdaQueryWrapper<AgentConversation>().eq(AgentConversation::getConversationId, conversationId));
        if (conv != null) {
            conv.setStatus(status);
            conv.setLastMessageTime(LocalDateTime.now());
            conversationMapper.updateById(conv);
        }
    }

    public AgentConversation getByConversationId(String conversationId) {
        return conversationMapper.selectOne(
                new LambdaQueryWrapper<AgentConversation>().eq(AgentConversation::getConversationId, conversationId));
    }

    public PageResult<AgentConversation> listByUser(Long userId, int pageNo, int pageSize) {
        return PageResult.of(conversationMapper.selectPage(new Page<>(pageNo, pageSize),
                new LambdaQueryWrapper<AgentConversation>()
                        .eq(AgentConversation::getUserId, userId)
                        .orderByDesc(AgentConversation::getUpdateTime)));
    }

    public PageResult<AgentConversation> listByUserAndCategory(Long userId, String category, int pageNo, int pageSize) {
        return PageResult.of(conversationMapper.selectPage(new Page<>(pageNo, pageSize),
                new LambdaQueryWrapper<AgentConversation>()
                        .eq(AgentConversation::getUserId, userId)
                        .eq(AgentConversation::getCategory, category)
                        .orderByDesc(AgentConversation::getUpdateTime)));
    }

    public List<AgentConversation> listByProject(Long projectId) {
        return conversationMapper.selectList(new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getProjectId, projectId)
                .orderByDesc(AgentConversation::getUpdateTime));
    }

    /**
     * 查询用户正在运行的 pipeline 对话（status=running）
     */
    public List<AgentConversation> listRunning(Long userId) {
        return conversationMapper.selectList(new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getUserId, userId)
                .eq(AgentConversation::getStatus, "running")
                .orderByDesc(AgentConversation::getUpdateTime));
    }

    public Mono<Void> deleteConversation(long id, long currentUserId) {
        return Mono.fromCallable(() -> conversationMapper.selectOne(ownedConversation(id, currentUserId)))
                .subscribeOn(schedulers.journal())
                .flatMap(conversation -> statePreflight.deleteConversationSessions(
                                String.valueOf(currentUserId), conversation.getConversationId())
                        .then(Mono.fromRunnable(() -> conversationMapper.delete(
                                        ownedConversation(id, currentUserId)))
                                .subscribeOn(schedulers.journal())))
                .then();
    }

    private LambdaQueryWrapper<AgentConversation> ownedConversation(long id, long userId) {
        return new LambdaQueryWrapper<AgentConversation>()
                .eq(AgentConversation::getId, id)
                .eq(AgentConversation::getUserId, userId);
    }
}
