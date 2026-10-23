package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * 使用同一图命令协议；仅把工具内的新建对象别名解析为稳定的真实 UUID。
 */
@Component
@RequiredArgsConstructor
public class CanvasAgentOperations {
    public record Prepared(CanvasEditCommand command, Map<String, String> ids) {
    }

    private static final Set<String> ID_FIELDS = Set.of("id", "nodeId", "sourceNodeId", "targetNodeId", "parentId",
            "edgeId");
    private final ObjectMapper mapper;

    public Prepared prepare(ArrayNode input, CanvasAgentAccess.Session session) {
        if (input == null || input.isEmpty() || input.size() > 100) {
            throw new BusinessException(400, "每批操作数量必须为 1 至 100");
        }
        ArrayNode operations = input.deepCopy();
        Map<String, String> ids = new LinkedHashMap<>();
        for (JsonNode operation : operations) {
            String type = operation.path("type").asText();
            JsonNode object = switch (type) {
                case "canvas.node.create" -> operation.path("node");
                case "canvas.edge.create" -> operation.path("edge");
                default -> null;
            };
            if (object == null) {
                continue;
            }
            String alias = object.path("id").asText();
            if (!alias.matches("\\$[A-Za-z0-9_-]{1,64}")) {
                throw new BusinessException(400, "新建节点或连线必须使用 $名称 作为本批别名");
            }
            if (ids.putIfAbsent(alias, session.id(alias)) != null) {
                throw new BusinessException(400, "新建对象别名不能重复");
            }
        }
        operations.forEach(operation -> replaceIds(operation, ids));
        ObjectNode command = mapper.createObjectNode().put("requestId", session.requestId());
        command.set("operations", operations);
        try {
            CanvasEditCommand parsed = mapper.readerFor(CanvasEditCommand.class)
                    .with(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES).readValue(command);
            return new Prepared(parsed, Map.copyOf(ids));
        } catch (IOException invalid) {
            throw new BusinessException(400, "图操作结构或字段无效，请按节点协议提交并携带预期版本");
        }
    }

    private void replaceIds(JsonNode node, Map<String, String> ids) {
        if (!(node instanceof ObjectNode object)) {
            return;
        }
        // 不进入 config/configPatch 或业务正文，避免改写用户文本和媒体标识。
        for (String field : ID_FIELDS) {
            JsonNode value = object.get(field);
            if (value != null && value.isTextual() && value.textValue().startsWith("$")) {
                String resolved = ids.get(value.textValue());
                if (resolved == null) {
                    throw new BusinessException(400, "引用的新建对象别名未在本批声明");
                }
                object.put(field, resolved);
            }
        }
        for (String nested : Set.of("node", "edge", "layout")) {
            if (object.has(nested)) {
                replaceIds(object.get(nested), ids);
            }
        }
    }
}
