package com.stonewu.fusion.service.canvas.run.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.networknt.schema.Schema;
import com.networknt.schema.SchemaRegistry;
import com.networknt.schema.SpecificationVersion;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;

/**
 * 契约不从远程地址装载 schema；执行参数和实际输出都走同一校验。
 */
@Component
public class CanvasMcpSchemas {
    private final SchemaRegistry registry = SchemaRegistry.withDefaultDialect(SpecificationVersion.DRAFT_2020_12,
            builder -> builder.schemaLoader(loader -> loader.fetchRemoteResources(false).block(ignored -> true)));

    public Schema compile(JsonNode definition) {
        if (definition == null || !definition.isObject()
                || definition.toString().getBytes(StandardCharsets.UTF_8).length > 128 * 1024) {
            throw CanvasRunException.invalid("MCP 契约需要不超过 128 KiB 的 JSON Schema 对象");
        }
        inspect(definition, 0);
        try {
            var schema = registry.getSchema(definition);
            schema.initializeValidators();
            return schema;
        } catch (RuntimeException failure) {
            throw CanvasRunException.invalid("MCP 的 JSON Schema 无效或引用无法解析");
        }
    }

    public void validate(JsonNode definition, JsonNode value, String label) {
        var errors = compile(definition).validate(value);
        if (!errors.isEmpty()) {
            throw CanvasRunException.invalid(label + "不符合 MCP 契约：" + errors.getFirst().getMessage());
        }
    }

    private void inspect(JsonNode value, int depth) {
        if (depth > 40) {
            throw CanvasRunException.invalid("MCP Schema 嵌套超过 40 层");
        }
        if (value.isObject()) {
            for (String key : new String[]{"$ref", "$dynamicRef", "$recursiveRef"}) {
                var ref = value.get(key);
                if (ref != null && (!ref.isTextual() || !ref.asText().startsWith("#"))) {
                    throw CanvasRunException.invalid("MCP Schema 只能引用同一契约内的定义");
                }
            }
            if (value.has("$id") || value.has("id")) {
                throw CanvasRunException.invalid("MCP Schema 不能重设外部标识");
            }
            var dialect = value.get("$schema");
            if (dialect != null && (!dialect.isTextual() || SpecificationVersion.fromDialectId(dialect.asText())
                    .isEmpty())) {
                throw CanvasRunException.invalid("MCP Schema 使用了不支持的规范版本");
            }
        }
        for (String keyword : new String[]{"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}) {
            var children = value.path(keyword);
            if (children.isObject()) {
                children.forEach(child -> inspect(child, depth + 1));
            }
        }
        for (String keyword : new String[]{"items", "prefixItems", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
                "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "contains", "propertyNames", "additionalItems"}) {
            var child = value.path(keyword);
            if (child.isArray()) {
                child.forEach(item -> inspect(item, depth + 1));
            } else if (child.isObject()) {
                inspect(child, depth + 1);
            }
        }
        var dependencies = value.path("dependencies");
        if (dependencies.isObject()) {
            dependencies.forEach(child -> {
                if (child.isObject()) {
                    inspect(child, depth + 1);
                }
            });
        }
    }
}
