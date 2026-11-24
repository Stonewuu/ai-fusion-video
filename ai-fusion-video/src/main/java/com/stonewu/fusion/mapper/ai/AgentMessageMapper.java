package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentMessageMapper extends BaseMapper<AgentMessage> {

    List<AgentMessage> selectConversationHistory(@Param("conversationId") String conversationId);

    List<AgentMessage> selectStreamHistory(@Param("streamRunId") String streamRunId,
                                         @Param("sourceRunId") String sourceRunId);

    List<AgentMessage> selectRootHistory(@Param("conversationId") String conversationId);

    Long selectInitialOrderByRunId(@Param("runId") String runId);

    AgentMessage selectByProjectionKey(@Param("projectionKey") String projectionKey);

    int insertMessages(@Param("messages") List<AgentMessage> messages);

    int cancelUnfinishedTools(
            @Param("runId") String runId,
            @Param("updateTime") LocalDateTime updateTime);
}
