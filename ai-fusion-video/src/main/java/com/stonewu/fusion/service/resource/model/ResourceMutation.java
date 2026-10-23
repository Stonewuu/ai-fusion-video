package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

public record ResourceMutation(ObjectNode result, List<ResourceChange> changes, ObjectNode previousValues) {
    public ResourceMutation {
        changes = List.copyOf(changes);
    }
}
