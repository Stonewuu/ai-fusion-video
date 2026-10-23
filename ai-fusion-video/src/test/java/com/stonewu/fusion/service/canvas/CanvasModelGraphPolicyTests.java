package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.command.CanvasGraphChangeSet;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import com.stonewu.fusion.service.canvas.model.CanvasOutputReference;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verify;

class CanvasModelGraphPolicyTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasModelCatalog models = mock(CanvasModelCatalog.class);
    private final CanvasModelGraphPolicy policy = new CanvasModelGraphPolicy(models, new CanvasCapabilityRegistry());
    private final CanvasGraph empty = new CanvasGraph(1, null, List.of(), List.of());

    @Test void rejectsDisabledModelsAndUnsupportedPortsWithoutProviderCalls() {
        var video = node("video", CanvasNodeKind.VIDEO_GENERATION, config().put("modelId", 9));
        when(models.require(9)).thenThrow(new BusinessException("模型停用"));
        assertThatThrownBy(() -> validate(empty, graph(List.of(video), List.of()), List.of(video), List.of())).isInstanceOf(CanvasException.class);
        doReturn(option("video", Map.of("prompt", input(1)), Map.of(), List.of())).when(models).require(9);
        var image = node("image", CanvasNodeKind.IMAGE, config());
        var edge = edge(image, "media", video, "firstFrame");
        assertThatCode(() -> validate(empty, graph(List.of(image,video),List.of(edge)),List.of(video),List.of(edge))).doesNotThrowAnyException();
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(image,video),List.of(edge)),Set.of()))
                .isInstanceOf(CanvasException.class).hasMessageContaining("首帧");
    }

    @Test void checksUpstreamCollectionCardinalityWhenTheSourceCountChanges() {
        when(models.require(9)).thenReturn(option("image", Map.of("referenceImages",input(3)), Map.of(),List.of()));
        var beforeSource = node("a",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9).put("count",2));
        var target = node("b",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9));
        var edge = edge(beforeSource,"images",target,"referenceImages");
        var before = graph(List.of(beforeSource,target),List.of(edge));
        var afterSource = node("a",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9).put("count",4));
        assertThatCode(() -> validate(before,graph(List.of(afterSource,target),List.of(edge)),List.of(afterSource),List.of())).doesNotThrowAnyException();
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(afterSource,target),List.of(edge)),Set.of()))
                .isInstanceOf(CanvasException.class).hasMessageContaining("最多允许 3");
    }

    @Test void rejectsMutuallyExclusiveVideoInputs() {
        when(models.require(9)).thenReturn(option("video",Map.of("firstFrame",input(1),"referenceImages",input(3)),Map.of("exclusiveInputModes",true),List.of()));
        var image = node("i",CanvasNodeKind.IMAGE,config());
        var video = node("v",CanvasNodeKind.VIDEO_GENERATION,config().put("modelId",9));
        var edges = List.of(edge(image,"media",video,"firstFrame"),edge(image,"media",video,"referenceImages"));
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(image,video),edges),Set.of()))
                .isInstanceOf(CanvasException.class).hasMessageContaining("互斥");
    }

    @Test void fixedImageInsideACollectionCountsAsOneForBothStoredAndImportedGraphs() {
        when(models.require(9)).thenReturn(option("image",Map.of("referenceImages",input(1)),Map.of(),List.of()));
        var source=node("source",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9).put("count",4));
        var collection=node("collection",CanvasNodeKind.COLLECTION,config().put("valueType","image"));
        var target=node("target",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9));
        var first=edge(source,"images",collection,"items");
        var last=edge(collection,"items",target,"referenceImages");
        var nodes=List.of(source,collection,target);
        assertThatThrownBy(() -> policy.validateAll(graph(nodes,List.of(first,last)),Set.of())).hasMessageContaining("最多允许 1");
        assertThatCode(() -> policy.validateAll(graph(nodes,List.of(first,last)),Set.of(first.id()))).doesNotThrowAnyException();
        var fixed=new CanvasEdge(first.id(),first.kind(),first.sourceNodeId(),first.sourcePort(),first.targetNodeId(),first.targetPort(),
                first.inputIndex(),new CanvasOutputReference("run","value",null),0);
        assertThatCode(() -> policy.validateAll(graph(nodes,List.of(fixed,last)),Set.of())).doesNotThrowAnyException();
    }

    @Test void requiresPublishedVersionBelongingToTheSelectedWorkflow() {
        var version = new CanvasModelCatalog.Version(31,2,Map.of("prompt",input(1)),List.of("prompt"), Map.of(), null);
        when(models.require(9)).thenReturn(new CanvasModelCatalog.Option(9,"测试工作流","comfyui","image",Map.of(),Map.of(),List.of(version),null));
        var invalid = node("c",CanvasNodeKind.COMFYUI,config().put("modelId",9).put("resultType","image").put("workflowVersionId",30));
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(invalid),List.of()),Set.of())).isInstanceOf(CanvasException.class);
        var valid = node("c",CanvasNodeKind.COMFYUI,config().put("modelId",9).put("resultType","image").put("workflowVersionId",31));
        assertThatCode(() -> policy.validateAll(graph(List.of(valid),List.of()),Set.of())).doesNotThrowAnyException();
    }

    @Test void disabledModelDoesNotPreventUnrelatedLayoutEdits() {
        var original = node("v",CanvasNodeKind.VIDEO_GENERATION,config().put("modelId",9));
        var moved = new CanvasNode(original.id(),original.kind(),1,"移动后的视图",new CanvasNodeLayout(40,80,null,null,null,false,false),original.config(),null,0,1);
        validate(graph(List.of(original),List.of()),graph(List.of(moved),List.of()),List.of(moved),List.of());
        verifyNoInteractions(models);
    }

    @Test void rejectsUnsupportedAspectRatiosAndDurationBounds() {
        when(models.require(9)).thenReturn(option("video", Map.of(),
                Map.of("supportedAspectRatios", List.of("16:9"), "minDuration", 3, "maxDuration", 10), List.of()));
        var wrongRatio = node("v", CanvasNodeKind.VIDEO_GENERATION, config().put("modelId", 9).put("ratio", "1:1"));
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(wrongRatio), List.of()),Set.of()))
                .isInstanceOf(CanvasException.class).hasMessageContaining("ratio");
        var tooLong = node("v", CanvasNodeKind.VIDEO_GENERATION, config().put("modelId", 9).put("duration", 11));
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(tooLong), List.of()),Set.of()))
                .isInstanceOf(CanvasException.class).hasMessageContaining("时长");
    }

    @Test void audioNodesRequireAnExplicitConfiguredService() {
        var missing = node("audio", CanvasNodeKind.MUSIC_GENERATION, config().put("operation", "music"));
        assertThatCode(() -> validate(empty, graph(List.of(missing), List.of()), List.of(missing), List.of())).doesNotThrowAnyException();
        assertThatThrownBy(() -> policy.validateAll(graph(List.of(missing), List.of()),Set.of()))
                .isInstanceOf(CanvasException.class).hasMessageContaining("音频服务");
        var selected = node("audio", CanvasNodeKind.MUSIC_GENERATION, config().put("operation", "music").put("apiConfigId", 12));
        validate(empty, graph(List.of(selected), List.of()), List.of(selected), List.of());
        verify(models).requireAudio(12);
    }

    @Test void unconfiguredModelsCanBePlacedButCannotRun() {
        var image = node("draft", CanvasNodeKind.IMAGE_GENERATION, config());
        validate(empty, graph(List.of(image),List.of()), List.of(image),List.of());
        verifyNoInteractions(models);
        assertThatThrownBy(() -> policy.validateForRun(graph(List.of(image),List.of()),image,Map.of()))
                .hasMessageContaining("选择模型");
    }

    @Test void reusedResultsUseTheirActualCountsInsteadOfCurrentGenerationSettings() {
        when(models.require(9)).thenReturn(option("image",Map.of("referenceImages",input(1)),Map.of(),List.of()));
        var source=node("source",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9).put("count",4));
        var collection=node("collection",CanvasNodeKind.COLLECTION,config().put("valueType","image"));
        var target=node("target",CanvasNodeKind.IMAGE_GENERATION,config().put("modelId",9));
        var graph=graph(List.of(source,collection,target),List.of(edge(source,"images",collection,"items"),edge(collection,"items",target,"referenceImages")));
        assertThatCode(() -> policy.validateForRun(graph,target,Map.of("source:images",1))).doesNotThrowAnyException();
        assertThatThrownBy(() -> policy.validateForRun(graph,target,Map.of("source:images",2))).hasMessageContaining("最多允许 1");
    }

    @Test void draftImportAllowsIncompleteAndIncompatibleInputsButVerifiesChosenProviderIdentity() {
        when(models.require(9)).thenReturn(option("image", Map.of("prompt", input(1)), Map.of(), List.of()));
        var source = node("source", CanvasNodeKind.IMAGE, config());
        var target = node("target", CanvasNodeKind.IMAGE_GENERATION, config().put("modelId", 9));
        var graph = graph(List.of(source, target), List.of(edge(source, "media", target, "referenceImages")));
        assertThatCode(() -> policy.validateDraft(graph)).doesNotThrowAnyException();
        assertThatThrownBy(() -> policy.validateForRun(graph, target, Map.of())).hasMessageContaining("参考图片");
        var wrongKind = node("video", CanvasNodeKind.VIDEO_GENERATION, config().put("modelId", 9));
        assertThatThrownBy(() -> policy.validateDraft(graph(List.of(wrongKind), List.of()))).hasMessageContaining("模型类型");
    }

    @Test void draftWorkflowBindingStillRequiresVersionAndOutputToBelongToSelectedModel() {
        var version = new CanvasModelCatalog.Version(31, 2, Map.of("prompt", input(1)), List.of("prompt"), Map.of(), null);
        when(models.require(9)).thenReturn(new CanvasModelCatalog.Option(9, "工作流", "comfyui", "image",
                Map.of(), Map.of(), List.of(version), null));
        var incomplete = node("draft", CanvasNodeKind.COMFYUI, config().put("modelId", 9));
        assertThatCode(() -> policy.validateDraft(graph(List.of(incomplete), List.of()))).doesNotThrowAnyException();
        var wrongVersion = node("wrong", CanvasNodeKind.COMFYUI, config().put("modelId", 9).put("workflowVersionId", 32));
        assertThatThrownBy(() -> policy.validateDraft(graph(List.of(wrongVersion), List.of()))).hasMessageContaining("已发布版本");
        var wrongOutput = node("wrong", CanvasNodeKind.COMFYUI, config().put("modelId", 9).put("resultType", "video"));
        assertThatThrownBy(() -> policy.validateDraft(graph(List.of(wrongOutput), List.of()))).hasMessageContaining("输出类型");
    }

    private CanvasModelCatalog.Option option(String kind,Map<String,CanvasModelCatalog.Input> inputs,Map<String,Object> parameters,List<CanvasModelCatalog.Version> versions) {
        return new CanvasModelCatalog.Option(9,"测试模型",kind,kind,inputs,parameters,versions,null);
    }
    private CanvasModelCatalog.Input input(int max) { return new CanvasModelCatalog.Input(max,0); }
    private ObjectNode config() { return json.createObjectNode(); }
    private CanvasNode node(String id,CanvasNodeKind kind,ObjectNode config) { return new CanvasNode(id,kind,1,id,new CanvasNodeLayout(0,0,null,null,null,false,false),config,null,0,0); }
    private CanvasEdge edge(CanvasNode source,String output,CanvasNode target,String input) { return new CanvasEdge(source.id()+":"+target.id()+":"+input,CanvasEdgeKind.EXECUTION,source.id(),output,target.id(),input,0,null,0); }
    private CanvasGraph graph(List<CanvasNode> nodes,List<CanvasEdge> edges) { return new CanvasGraph(1,null,nodes,edges); }
    private void validate(CanvasGraph before,CanvasGraph after,List<CanvasNode> nodes,List<CanvasEdge> edges) {
        policy.validate(null,before,new CanvasGraphMutation(after,new CanvasGraphChangeSet(nodes,List.of(),edges,List.of()),new CanvasGraphChangeSet(List.of(),List.of(),List.of(),List.of())),null);
    }
}
