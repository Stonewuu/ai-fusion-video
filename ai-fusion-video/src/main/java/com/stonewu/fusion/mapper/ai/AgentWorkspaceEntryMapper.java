package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface AgentWorkspaceEntryMapper extends BaseMapper<AgentWorkspaceEntry> {

    @Select("""
            SELECT * FROM afv_agent_workspace_entry
            WHERE namespace_hash = #{namespaceHash} AND item_hash = #{itemHash} AND deleted = 0
            LIMIT 1 FOR UPDATE
            """)
    AgentWorkspaceEntry selectForUpdate(
            @Param("namespaceHash") String namespaceHash,
            @Param("itemHash") String itemHash);

    @Delete("DELETE FROM afv_agent_workspace_entry WHERE id = #{id}")
    int physicalDeleteById(@Param("id") Long id);

    @Select("SELECT COALESCE(SUM(content_size), 0) FROM afv_agent_workspace_entry WHERE deleted = 0")
    long sumContentSize();
}
