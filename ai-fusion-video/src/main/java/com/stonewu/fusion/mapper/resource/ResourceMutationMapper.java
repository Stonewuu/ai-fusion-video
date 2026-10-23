package com.stonewu.fusion.mapper.resource;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * 多类型资源命令的字段集合由各业务字段白名单构建，不接受外部表名或 SQL。
 */
@Mapper
public interface ResourceMutationMapper {
    @Data
    @RequiredArgsConstructor
    class NewResource {
        private Long id;
        private final CanvasValueType type;
        private final Map<String, Object> columns;
    }

    int insert(@Param("row") NewResource row);

    int insertCopy(@Param("row") NewResource row);

    int patch(@Param("type") CanvasValueType type, @Param("id") long id, @Param("version") int version,
              @Param("columns") Map<String, Object> columns, @Param("includeDeleted") boolean includeDeleted);

    int replaceCopy(@Param("type") CanvasValueType type, @Param("id") long id,
                    @Param("columns") Map<String, Object> columns);

    int markDeletedEpisodeCopy(@Param("id") long id);

    List<Long> lockChildren(@Param("type") CanvasValueType type, @Param("column") String column,
                            @Param("parentId") long parentId);

    List<Long> lockEpisodeShots(@Param("episodeId") long episodeId);

    List<Map<String, Object>> lockProjectTitles(@Param("type") CanvasValueType type,
                                                @Param("projectId") long projectId);

    List<String> columns(@Param("table") String table);

    List<Long> referencing(@Param("type") CanvasValueType type, @Param("column") String column,
                           @Param("collection") boolean collection, @Param("nestedPath") String nestedPath,
                           @Param("resourceId") long resourceId, @Param("jsonId") String jsonId);
}
