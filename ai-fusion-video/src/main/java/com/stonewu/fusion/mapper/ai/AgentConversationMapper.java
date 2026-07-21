package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentConversation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface AgentConversationMapper extends BaseMapper<AgentConversation> {

    @Select("""
            SELECT *
            FROM afv_agent_conversation
            WHERE conversation_id = #{conversationId}
              AND deleted = 0
            LIMIT 1
            FOR UPDATE
            """)
    AgentConversation selectByConversationIdForUpdate(
            @Param("conversationId") String conversationId);
}
