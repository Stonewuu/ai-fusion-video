package com.stonewu.fusion.mapper.media;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaDeletionEntity;
import com.stonewu.fusion.service.media.deletion.MediaDeletionData.Reference;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface MediaDeletionMapper extends BaseMapper<MediaDeletionEntity> {
    List<Reference> references(@Param("mediaId") String mediaId);

    boolean activeUse(@Param("mediaId") String mediaId);

    List<String> previewOutputs(@Param("mediaId") String mediaId);

    long sharedCopies(@Param("mediaId") String mediaId, @Param("blobId") String blobId);
}
