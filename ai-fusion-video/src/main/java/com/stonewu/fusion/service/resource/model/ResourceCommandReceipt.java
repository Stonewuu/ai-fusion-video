package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

public record ResourceCommandReceipt(String commandId, String requestId, ObjectNode result,
                                     List<ResourceChange> changes) {
    public ResourceCommandReceipt {
        changes = List.copyOf(changes);
    }
}
