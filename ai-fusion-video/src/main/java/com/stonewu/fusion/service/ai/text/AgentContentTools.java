package com.stonewu.fusion.service.ai.text;

import com.stonewu.fusion.service.ai.tool.ReadToolResultToolExecutor;
import com.stonewu.fusion.service.ai.tool.script.ReadScriptContentToolExecutor;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;

/** 工具返回分段读取入口时，相应读取工具属于必需依赖。 */
public final class AgentContentTools {
    private static final Set<String> SCRIPT_QUERIES = Set.of("get_script", "get_project_script",
            "get_script_structure", "get_script_episode");
    public static final String INSTRUCTION = """
            ## 长内容读取
            - 工具返回 readTool 时，使用指定的读取工具，不猜测其他工具名称，也不把读取入口当作正文。
            - contentReads 提供可选的原文入口，仅在当前任务需要原文时使用。生成分镜和预处理分镜资产默认使用已解析场次、动作和对白，不因出现原文入口就重新读取小说。
            - limit 可省略，由平台按当前模型配置的上下文容量分配；需要较短内容时再主动减小，不要固定请求 8000 字符。
            - 从 offset=0 开始，携带 contentVersion，按每段返回的 nextOffset 连续读取，直到 hasMore=false。
            - 按章节或场次记录已处理的范围和下一段位置。分页边界不代表章节边界，跨段的章节和场次需要接着读取。
            - 需要完整解析时不得遗漏中间内容，只有覆盖全部要求范围后才能报告完成；不要要求用户重新粘贴已保存的原文。
            - contentVersion 标识正文内容，与写入时使用的 expectedVersion 不同，不可互换。
            - 原文版本变化或读取失败时说明具体原因，不使用首尾片段推测缺失内容，不把未读取部分标记为完成。
            """;

    private AgentContentTools() {
    }

    public static Set<String> required(Collection<String> names) {
        var required = new LinkedHashSet<String>();
        if (!names.isEmpty()) {
            required.add(ReadToolResultToolExecutor.NAME);
        }
        if (names.stream().anyMatch(SCRIPT_QUERIES::contains)) {
            required.add(ReadScriptContentToolExecutor.NAME);
        }
        return required;
    }

    public static String schema(String name) {
        return switch (name) {
            case ReadToolResultToolExecutor.NAME -> ReadToolResultToolExecutor.SCHEMA;
            case ReadScriptContentToolExecutor.NAME -> ReadScriptContentToolExecutor.SCHEMA;
            default -> throw new IllegalArgumentException("未登记的内容读取工具：" + name);
        };
    }
}
