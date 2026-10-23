package com.stonewu.fusion.mapper.ai;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentModelRequestEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface AgentModelRequestMapper extends BaseMapper<AgentModelRequestEntity> {
    long uncertainForConversation(@Param("conversationId") String conversationId);

    long countHeldForModel(@Param("modelId") long modelId);

    int admit(@Param("id") String id, @Param("runId") String runId, @Param("modelId") long modelId,
              @Param("owner") String owner, @Param("epoch") long epoch);

    int completed(@Param("id") String id);

    int uncertain(@Param("id") String id);

    String lockRequest(@Param("id") String id, @Param("runId") String runId, @Param("owner") String owner,
                       @Param("epoch") long epoch);

    int reconcile();

    int adminCancel(@Param("id") String id);
}
