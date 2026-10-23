package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;

public record ResourceSnapshot(ResourceHeader header, ObjectNode data) {
}
