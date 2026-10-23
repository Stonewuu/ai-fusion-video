package com.stonewu.fusion.mapper.generation;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.generation.GenerationAttemptEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface GenerationAttemptMapper extends BaseMapper<GenerationAttemptEntity> {
    int nextNumber(@Param("jobId") String jobId);

    int insertPrepared(@Param("id") String id, @Param("jobId") String jobId, @Param("number") int number);

    int beginSubmission(@Param("request") String request, @Param("providerTaskId") String providerTaskId,
                        @Param("id") String id);

    List<LocalDateTime> lockRetriedAcknowledgements(@Param("jobId") String jobId);

    String lock(@Param("id") String id);

    int acknowledge(@Param("providerTaskId") String providerTaskId, @Param("result") String result,
                    @Param("id") String id);

    List<String> lockHistory(@Param("jobId") String jobId);

    int resolve(@Param("action") String action, @Param("note") String note, @Param("userId") long userId,
                @Param("id") String id);

    int adminResolve(@Param("note") String note, @Param("admin") long admin, @Param("id") String id);
}
