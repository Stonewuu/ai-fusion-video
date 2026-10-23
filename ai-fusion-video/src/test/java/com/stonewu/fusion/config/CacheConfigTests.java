package com.stonewu.fusion.config;

import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import com.stonewu.fusion.service.ai.catalog.ProviderCatalogService;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpWorkflowTool;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Contract;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Definition;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Effect;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Idempotency;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.OutputMode;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class CacheConfigTests {

    @Test
    void serializerRoundTripsImmutableAndConcreteCatalogTypes() throws Exception {
        GenericJackson2JsonRedisSerializer serializer = CacheConfig.valueSerializer();
        List<AgentUserSkillService.UserSkillSummary> immutable = List.of(
                new AgentUserSkillService.UserSkillSummary(
                        "story-review_workspace:user",
                        "story-review",
                        "故事结构检查",
                        "检查故事结构",
                        "workspace:user"));
        ArrayList<AgentUserSkillService.UserSkillSummary> concrete = new ArrayList<>(immutable);

        assertThat(roundTrip(serializer, immutable)).isEqualTo(immutable);
        assertThat(roundTrip(serializer, concrete)).isEqualTo(concrete);
        var providers = new ProviderCatalogService(new ObjectMapper()).list();
        assertThat(roundTrip(serializer, providers)).isEqualTo(providers);
    }

    @Test
    void serializerRoundTripsStoryboardItemsWithDecimalDuration() {
        GenericJackson2JsonRedisSerializer serializer = CacheConfig.valueSerializer();
        List<StoryboardItem> items = List.of(StoryboardItem.builder()
                .id(1L)
                .storyboardId(3L)
                .duration(new BigDecimal("3.5"))
                .build());

        assertThat(roundTrip(serializer, items)).isEqualTo(items);
    }

    @Test
    void cacheKeysCarryAnExplicitSerializationSchemaVersion() {
        assertThat(CacheConfig.CACHE_SCHEMA_PREFIX).isEqualTo("afv:cache:v3:");
    }

    @Test void mcpContractCachePreservesJsonSchemasAndTreatsTypeLikeFieldsAsOrdinaryContent() throws Exception {
        var mapper=new ObjectMapper();
        var schema=mapper.readTree("{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"},\"@class\":{\"const\":\"example.Type\"}},\"required\":[\"text\"],\"additionalProperties\":false}");
        var output=mapper.readTree("{\"type\":\"string\"}");
        var tool=new McpWorkflowTool("本机","inspect","说明",schema,output,true,true,false,"configuration","schema");
        var value=new Contract("contract",9,"读取",2,true,new Definition(tool,Effect.READ_ONLY,Idempotency.READ_ONLY,"",OutputMode.TEXT,output,30),"hash");
        var serializer=CacheConfig.mcpContractSerializer();
        assertThat(serializer.deserialize(serializer.serialize(value))).isEqualTo(value);
        assertThat(mapper.readTree(serializer.serialize(value)).has("@class")).isFalse();
    }

    private Object roundTrip(
            GenericJackson2JsonRedisSerializer serializer,
            Object value) {
        return serializer.deserialize(serializer.serialize(value));
    }
}
