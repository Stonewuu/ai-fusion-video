package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeModels;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScopes;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.command.CanvasGraphChangeSet;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import com.stonewu.fusion.service.canvas.run.CanvasRunEnvironment;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextModels;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class CanvasDraftReadinessTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasCapabilityRegistry registry = new CanvasCapabilityRegistry();
    private final CanvasConfigValidator configs = new CanvasConfigValidator();
    private final CanvasModelCatalog models = mock(CanvasModelCatalog.class);
    private final GenerationModelSnapshot providers = mock(GenerationModelSnapshot.class);
    private final CanvasRunEnvironment environment = new CanvasRunEnvironment(mock(CanvasJson.class), models, providers,
            mock(CanvasTextModels.class), mock(CanvasAgentNodeModels.class), mock(CanvasAgentNodeScopes.class),
            mock(CanvasMcpContracts.class), registry, configs, new CanvasModelGraphPolicy(models, registry));

    @Test void unboundProjectContentAndMediaAreDraftsButNotRunnable() {
        for (var kind : List.of(CanvasNodeKind.SHOT, CanvasNodeKind.IMAGE)) {
            var node = node(kind, json.createObjectNode());
            var graph = new CanvasGraph(1, 1L, List.of(node), List.of());
            assertThatCode(() -> new CanvasGraphValidator(registry, configs).validate(graph)).doesNotThrowAnyException();
            assertThatThrownBy(() -> environment.validateForRun(graph, node, Map.of()))
                    .isInstanceOf(BusinessException.class);
        }
        verifyNoInteractions(models, providers);
    }

    @Test void missingModelStopsBeforeFreezingAnyProvider() {
        var node = node(CanvasNodeKind.IMAGE_GENERATION, json.createObjectNode().put("prompt", "雨夜"));
        assertThatThrownBy(() -> environment.validateForRun(new CanvasGraph(1,1L,List.of(node),List.of()),node,Map.of()))
                .hasMessageContaining("选择模型");
        verifyNoInteractions(models, providers);
    }

    @Test void draftValidationStillRejectsUnknownFieldsAndInvalidValues() {
        var capability = registry.require(CanvasNodeKind.IMAGE_GENERATION);
        assertThatThrownBy(() -> configs.validateDraft(capability,json.createObjectNode().put("unknown",true)))
                .isInstanceOf(CanvasException.class);
        assertThatThrownBy(() -> configs.validateDraft(capability,json.createObjectNode().put("count",-1)))
                .isInstanceOf(CanvasException.class);
    }

    @Test void emptyMediaDraftDoesNotSkipAuthorizationWhenAFileIsSelected() {
        var media = mock(MediaResourceQueries.class);
        var access = mock(MediaAccessPolicy.class);
        var policy = new CanvasMediaGraphPolicy(media, access);
        var empty = new CanvasGraph(1,1L,List.of(),List.of());
        var draft = node(CanvasNodeKind.IMAGE,json.createObjectNode());
        policy.validate(null,empty,mutation(draft),null);
        verifyNoInteractions(media, access);
        String mediaId = UUID.randomUUID().toString();
        var configured = node(CanvasNodeKind.IMAGE,json.createObjectNode().put("resourceId",mediaId));
        when(media.get(mediaId,null)).thenThrow(new BusinessException(403,"无权查看此媒体"));
        assertThatThrownBy(() -> policy.validate(null,empty,mutation(configured),null)).hasMessageContaining("无权");
        verifyNoInteractions(access);
    }

    private CanvasGraphMutation mutation(CanvasNode node) {
        var changes = new CanvasGraphChangeSet(List.of(node),List.of(),List.of(),List.of());
        return new CanvasGraphMutation(new CanvasGraph(1,1L,List.of(node),List.of()),changes,changes);
    }
    private CanvasNode node(CanvasNodeKind kind,ObjectNode config) {
        return new CanvasNode(UUID.randomUUID().toString(),kind,1,"待配置节点",new CanvasNodeLayout(0,0,null,null,null,false,false),config,null,0,0);
    }
}
