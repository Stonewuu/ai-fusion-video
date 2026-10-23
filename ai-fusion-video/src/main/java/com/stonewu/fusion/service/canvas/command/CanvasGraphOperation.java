package com.stonewu.fusion.service.canvas.command;

import com.fasterxml.jackson.annotation.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = CanvasGraphOperation.AssertNode.class, name = "canvas.node.assert"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.CreateNode.class, name = "canvas.node.create"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.PatchNode.class, name = "canvas.node.patch"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.LayoutNode.class, name = "canvas.node.layout"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.BindResource.class, name = "canvas.node.bind"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.DeleteNode.class, name = "canvas.node.delete"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.CreateEdge.class, name = "canvas.edge.create"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.ReplaceEdge.class, name = "canvas.edge.replace"),
        @JsonSubTypes.Type(value = CanvasGraphOperation.DeleteEdge.class, name = "canvas.edge.delete")
})
public sealed interface CanvasGraphOperation {
    record AssertNode(String nodeId,
                      @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedConfigVersion)
            implements CanvasGraphOperation {
    }

    record CreateNode(CanvasNode node) implements CanvasGraphOperation {
    }

    record PatchNode(String nodeId,
                     @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedConfigVersion,
                     String title,
                     ObjectNode configPatch) implements CanvasGraphOperation {
    }

    record LayoutNode(String nodeId,
                      @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedLayoutVersion,
                      CanvasNodeLayout layout) implements CanvasGraphOperation {
    }

    record BindResource(String nodeId,
                        @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedConfigVersion,
                        CanvasResourceReference resource) implements CanvasGraphOperation {
    }

    record DeleteNode(String nodeId,
                      @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedConfigVersion,
                      @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedLayoutVersion) implements CanvasGraphOperation {
    }

    record CreateEdge(CanvasEdge edge) implements CanvasGraphOperation {
    }

    record ReplaceEdge(CanvasEdge edge,
                       @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedVersion) implements CanvasGraphOperation {
    }

    record DeleteEdge(String edgeId,
                      @JsonProperty(required = true) @JsonSetter(nulls = Nulls.FAIL) long expectedVersion) implements CanvasGraphOperation {
    }
}
