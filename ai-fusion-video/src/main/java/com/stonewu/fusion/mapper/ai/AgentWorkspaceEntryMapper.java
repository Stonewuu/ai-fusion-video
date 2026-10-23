package com.stonewu.fusion.mapper.ai;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface AgentWorkspaceEntryMapper extends BaseMapper<AgentWorkspaceEntry> {

    AgentWorkspaceEntry selectForUpdate(
            @Param("namespaceHash") String namespaceHash,
            @Param("itemHash") String itemHash);

    int physicalDeleteById(@Param("id") Long id);

    long sumContentSize();

    int indexMediaReferences(@Param("references") String references, @Param("id") long id,
                             @Param("version") Long version, @Param("hash") String hash);

    List<AgentWorkspaceEntry> window(@Param("namespaceHash") String namespaceHash, @Param("limit") int limit,
                                     @Param("offset") int offset);
}
