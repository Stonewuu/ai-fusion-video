package com.stonewu.fusion.config;

import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class CacheConfigTests {

    @Test
    void serializerRoundTripsImmutableAndConcreteCatalogTypes() {
        GenericJackson2JsonRedisSerializer serializer = CacheConfig.valueSerializer();
        List<AgentUserSkillService.UserSkillSummary> immutable = List.of(
                new AgentUserSkillService.UserSkillSummary(
                        "story-review_workspace:user",
                        "story-review",
                        "检查故事结构",
                        "workspace:user"));
        ArrayList<AgentUserSkillService.UserSkillSummary> concrete = new ArrayList<>(immutable);

        assertThat(roundTrip(serializer, immutable)).isEqualTo(immutable);
        assertThat(roundTrip(serializer, concrete)).isEqualTo(concrete);
    }

    @Test
    void cacheKeysCarryAnExplicitSerializationSchemaVersion() {
        assertThat(CacheConfig.CACHE_SCHEMA_PREFIX).isEqualTo("afv:cache:v2:");
    }

    private Object roundTrip(
            GenericJackson2JsonRedisSerializer serializer,
            Object value) {
        return serializer.deserialize(serializer.serialize(value));
    }
}
