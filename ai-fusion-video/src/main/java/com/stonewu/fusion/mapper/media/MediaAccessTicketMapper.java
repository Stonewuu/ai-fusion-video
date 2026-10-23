package com.stonewu.fusion.mapper.media;


import java.time.LocalDateTime;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaAccessTicketEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MediaAccessTicketMapper extends BaseMapper<MediaAccessTicketEntity> {
    int removeExpired();

    LocalDateTime agentExpiry(@Param("runId") String runId, @Param("userId") long userId, @Param("teamId") Long teamId,
                              @Param("owner") String owner, @Param("epoch") long epoch,
                              @Param("projectId") Long projectId);

    LocalDateTime textExpiry(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch,
                             @Param("userId") long userId, @Param("teamId") Long teamId,
                             @Param("projectId") Long projectId);

    MediaAccessTicketEntity authorize(@Param("hash") String hash, @Param("id") String id);

    long validGeneration(@Param("id") String id, @Param("attemptId") String attemptId, @Param("userId") long userId,
                         @Param("teamId") Long teamId, @Param("projectId") Long projectId);

    long validAgent(@Param("runId") String runId, @Param("userId") long userId, @Param("teamId") Long teamId,
                    @Param("owner") String owner, @Param("epoch") Long epoch, @Param("projectId") Long projectId);

    long validText(@Param("id") String id, @Param("userId") long userId, @Param("teamId") Long teamId,
                   @Param("projectId") Long projectId);
}
