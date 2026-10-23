package com.stonewu.fusion.mapper.generation;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.generation.AsyncOutboxEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface AsyncOutboxMapper extends BaseMapper<AsyncOutboxEntity> {
    int enqueue(@Param("key") String key, @Param("type") String type,
                @Param("aggregate") String aggregate, @Param("payload") String payload);

    AsyncOutboxEntity nextPending();

    int backoff(@Param("id") long id, @Param("error") String error);

    int delivered(@Param("id") long id);

    int appendEvent(@Param("stream") String stream, @Param("outboxId") long outboxId,
                    @Param("aggregate") String aggregate, @Param("revision") long revision,
                    @Param("type") String type, @Param("payload") String payload);
}
