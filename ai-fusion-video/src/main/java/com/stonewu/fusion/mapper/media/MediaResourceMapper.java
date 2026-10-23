package com.stonewu.fusion.mapper.media;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaResourceEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface MediaResourceMapper extends BaseMapper<MediaResourceEntity> {
    MediaResourceEntity lockRequest(@Param("actor") long actor, @Param("requestId") String requestId);

    MediaResourceEntity lock(@Param("id") String id);

    MediaResourceEntity source(@Param("type") String type, @Param("id") String id, @Param("version") int version,
                               @Param("field") String field, @Param("lock") boolean lock);

    String directRetention(@Param("mediaId") String mediaId);

    String documentRetention(@Param("mediaId") String mediaId);

    List<MediaResourceEntity> readyResources(@Param("projectId") Long projectId, @Param("ownerType") int ownerType,
                                             @Param("ownerId") long ownerId, @Param("limit") int limit,
                                             @Param("offset") int offset);

    int copy(@Param("id") String id, @Param("projectId") Long projectId, @Param("ownerType") int ownerType,
             @Param("ownerId") long ownerId, @Param("actorUserId") long actorUserId,
             @Param("actorTeamId") Long actorTeamId, @Param("requestId") String requestId,
             @Param("requestHash") String requestHash, @Param("sourceId") String sourceId);

    int recordCopy(@Param("id") String id, @Param("sourceId") String sourceId, @Param("version") int version,
                   @Param("actor") long actor);

    int expireIngest(@Param("id") String id);

    int restartIngest(@Param("blobId") String blobId, @Param("epoch") long epoch, @Param("id") String id);

    int finishIngest(@Param("id") String id);

    int failIngest(@Param("id") String id);

    List<String> lockExpiredIngests();

    int insertStaging(@Param("resource") MediaResourceEntity resource);
}
