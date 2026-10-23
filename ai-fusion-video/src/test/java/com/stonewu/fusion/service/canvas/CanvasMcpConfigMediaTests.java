package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import org.junit.jupiter.api.Test;
import java.util.HashSet;
import java.util.Map;
import java.util.UUID;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CanvasMcpConfigMediaTests {
    private final CanvasJson json=new CanvasJson(new ObjectMapper());
    private final CanvasMcpConfigMedia media=new CanvasMcpConfigMedia(json);
    @Test void rewritesOnlyExplicitUrisInsideMcpJsonAndPreservesForeignIdentifiers() {
        String original=UUID.randomUUID().toString(),other=UUID.randomUUID().toString();
        var args=Map.of("modelId",7,"resourceId",original,"nested",Map.of("file","media://"+original));
        var node=node(CanvasNodeKind.MCP_TOOL,Map.of("inputMode","NONE","argumentsJson",json.write(args)));
        var ids=new HashSet<String>(); media.collect(node,ids); assertThat(ids).containsExactly(original);
        var copy=json.read(media.rewrite(node,id->"media://"+other).path("argumentsJson").asText(),ObjectNode.class);
        assertThat(copy.path("modelId").asInt()).isEqualTo(7);
        assertThat(copy.path("resourceId").asText()).isEqualTo(original);
        assertThat(copy.at("/nested/file").asText()).isEqualTo("media://"+other);
        assertThat(node.config().path("argumentsJson").asText()).isEqualTo(json.write(args));
    }
    @Test void handlesBatchMcpAndFixedJsonInputWithoutTreatingOrdinaryTextAsParameters() {
        String id=UUID.randomUUID().toString();
        var config=Map.of("inputMode","JSON_OBJECT","input",json.write(Map.of("file","media://"+id)),"argumentsJson","{}");
        var batch=node(CanvasNodeKind.FOREACH,Map.of("operationKind","mcpTool","operationConfig",config));
        var ids=new HashSet<String>();media.collect(batch,ids);assertThat(ids).containsExactly(id);
        ids.clear();media.collect(node(CanvasNodeKind.TEXT,Map.of("content",json.write(config))),ids);assertThat(ids).isEmpty();
    }
    @Test void malformedMcpParametersRequireCorrectionBeforeExportOrCopy() {
        assertThatThrownBy(()->media.collect(node(CanvasNodeKind.MCP_TOOL,Map.of("argumentsJson","[1,2]")),new HashSet<>()))
                .isInstanceOf(BusinessException.class).hasMessageContaining("JSON 对象");
    }
    private CanvasNode node(CanvasNodeKind kind,Map<String,Object> config) {
        return new CanvasNode(UUID.randomUUID().toString(),kind,1,"测试",null,json.object(config),null,0,0);
    }
}
