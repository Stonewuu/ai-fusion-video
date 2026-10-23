package com.stonewu.fusion.service.ai.configuration;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.ai.AgentUserOverrideEntity;

import com.stonewu.fusion.common.BusinessException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.dao.DuplicateKeyException;
import com.stonewu.fusion.mapper.ai.AgentUserOverrideMapper;
import org.springframework.stereotype.Repository;

@Repository
@RequiredArgsConstructor
public class AgentUserOverrideRepository {
    public static final String CACHE = "agentUserOverrides";

    public record Stored(long revision, String json) {
    }

    private final AgentUserOverrideMapper overrides;

    @Cacheable(value = CACHE, key = "#userId + ':' + #type")
    public Stored get(long userId, String type) {
        var row = overrides.selectOne(Wrappers.<AgentUserOverrideEntity>lambdaQuery()
                .eq(AgentUserOverrideEntity::getUserId, userId).eq(AgentUserOverrideEntity::getAgentType, type));
        return row == null ? new Stored(0, "{}") : new Stored(row.getRevision(), row.getOverridesJson());
    }

    public long save(long userId, String type, long expectedRevision, String json) {
        if (expectedRevision == 0) {
            try {
                var row = new AgentUserOverrideEntity();
                row.setUserId(userId);
                row.setAgentType(type);
                row.setOverridesJson(json);
                row.setRevision(1L);
                overrides.insert(row);
            } catch (DuplicateKeyException conflict) {
                throw changed();
            }
            return 1;
        }
        int updated = overrides.updateRevision(json, userId, type, expectedRevision);
        if (updated != 1) {
            throw changed();
        }
        return expectedRevision + 1;
    }

    private BusinessException changed() {
        return new BusinessException(409, "这份智能体配置已在其他窗口修改，请刷新后再保存");
    }
}
