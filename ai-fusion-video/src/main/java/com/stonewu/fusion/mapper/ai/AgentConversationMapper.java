package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentConversation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentConversationMapper extends BaseMapper<AgentConversation> {

    AgentConversation selectByConversationIdForUpdate(
            @Param("conversationId") String conversationId);

    LocalDateTime selectDatabaseNow();

    List<AgentConversation> selectAgentStateCleanupCandidates(
            @Param("cutoff") LocalDateTime cutoff,
            @Param("limit") int limit);

    int markAgentStateExpired(
            @Param("id") long id,
            @Param("expectedLastActiveAt") LocalDateTime expectedLastActiveAt,
            @Param("expiredAt") LocalDateTime expiredAt);
}
