package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;

import java.util.*;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.*;

/**
 * 工具直接声明图命令的字段，模型不需要猜测 type/op 或节点类型。
 */
final class CanvasAgentGraphSchema {
    private CanvasAgentGraphSchema() {
    }

    static ObjectNode kind() {
        return choice("已登记的节点类型；便签使用 note，annotation 仅为连线类型",
                Arrays.stream(CanvasNodeKind.values()).map(CanvasNodeKind::code).toArray(String[]::new));
    }

    static ObjectNode operation() {
        var result = object("图命令必须严格选择一种结构；每种命令只接受自己声明的字段");
        var variants = result.putArray("oneOf");
        var nodeId = string("已存在的节点 ID 或本批创建别名");
        var configVersion = integer("刚读取的节点 configVersion");
        var layoutVersion = integer("刚读取的节点 layoutVersion");
        variants.add(command("canvas.node.create", Map.of("node", node()), "node"));
        variants.add(command("canvas.node.assert", Map.of("nodeId", nodeId, "expectedConfigVersion", configVersion),
                "nodeId", "expectedConfigVersion"));
        variants.add(command("canvas.node.patch", Map.of("nodeId", nodeId, "expectedConfigVersion", configVersion,
                        "title", string("新标题"), "configPatch", object("只包含要修改的配置字段")), "nodeId",
                "expectedConfigVersion"));
        variants.add(command("canvas.node.layout", Map.of("nodeId", nodeId, "expectedLayoutVersion", layoutVersion,
                "layout", layout()), "nodeId", "expectedLayoutVersion", "layout"));
        variants.add(command("canvas.node.bind", Map.of("nodeId", nodeId, "expectedConfigVersion", configVersion,
                "resource", nullable(resource())), "nodeId", "expectedConfigVersion", "resource"));
        variants.add(command("canvas.node.delete", Map.of("nodeId", nodeId, "expectedConfigVersion", configVersion,
                "expectedLayoutVersion", layoutVersion), "nodeId", "expectedConfigVersion", "expectedLayoutVersion"));
        variants.add(command("canvas.edge.create", Map.of("edge", edge()), "edge"));
        variants.add(command("canvas.edge.replace",
                Map.of("edge", edge(), "expectedVersion", integer("刚读取的连线 version")),
                "edge", "expectedVersion"));
        variants.add(command("canvas.edge.delete",
                Map.of("edgeId", string("要删除的连线 ID"), "expectedVersion", integer("刚读取的连线 version")),
                "edgeId", "expectedVersion"));
        return result;
    }

    static ObjectNode operations() {
        return array(operation(), 100).put("minItems", 1);
    }

    private static ObjectNode command(String type, Map<String, ObjectNode> fields, String... required) {
        var properties = new LinkedHashMap<>(fields);
        properties.put("type", choice("命令类型，字段名必须为 type", type));
        var names = new ArrayList<>(List.of(required));
        names.addFirst("type");
        return schema(properties, names.toArray(String[]::new));
    }

    private static ObjectNode node() {
        return schema(Map.ofEntries(
                        Map.entry("id", string("新建节点必须用 $名称 别名")), Map.entry("kind", kind()),
                        Map.entry("schemaVersion", integer("固定为 1")), Map.entry("title", string("节点标题")),
                        Map.entry("config", object("按能力目录填写；text/note 的正文在 content 字段")),
                        Map.entry("layout", layout()), Map.entry("resource", nullable(resource())),
                        Map.entry("configVersion", integer("新建固定为 0")),
                        Map.entry("layoutVersion", integer("新建固定为 0"))),
                "id", "kind", "schemaVersion", "title", "config", "layout", "configVersion", "layoutVersion");
    }

    private static ObjectNode layout() {
        return schema(Map.of("x", number("横坐标"), "y", number("纵坐标"),
                        "width", nullable(number("节点宽度")), "height", nullable(number("分组高度")),
                        "parentId", nullable(string("所在分组 ID")), "locked", bool("是否锁定位置"), "collapsed",
                        bool("是否折叠分组")),
                "x", "y", "locked", "collapsed");
    }

    private static ObjectNode resource() {
        return schema(Map.of("type", string("查询获得的业务类型"), "id", integer("真实业务编号"), "version",
                        integer("真实业务版本")),
                "type", "id", "version");
    }

    private static ObjectNode edge() {
        return schema(Map.ofEntries(
                        Map.entry("id", string("create 使用 $名称 别名；replace 必须在 edge.id 中填写已有真实连线 ID")),
                        Map.entry("kind",
                                choice("annotation=注释线；reference=来源引用；execution=执行依赖", "execution", "reference",
                                        "annotation")),
                        Map.entry("sourceNodeId", string("来源节点 ID 或本批别名")),
                        Map.entry("targetNodeId", string("目标节点 ID 或本批别名")),
                        Map.entry("sourcePort", nullable(string("执行连线的真实输出端口"))),
                        Map.entry("targetPort", nullable(string("执行连线的真实输入端口"))),
                        Map.entry("inputIndex", nullable(integer("执行连线输入序号，从 0 开始"))),
                        Map.entry("output", nullable(object("仅显式固定已有输出时提供查询获得的完整引用"))),
                        Map.entry("version", integer("create 固定为 0；replace 保留刚读取的 version"))),
                "id", "kind", "sourceNodeId", "targetNodeId", "version");
    }

    private static ObjectNode nullable(ObjectNode value) {
        var result = JsonNodeFactory.instance.objectNode();
        result.putArray("anyOf").add(value).add(JsonNodeFactory.instance.objectNode().put("type", "null"));
        return result;
    }
}
