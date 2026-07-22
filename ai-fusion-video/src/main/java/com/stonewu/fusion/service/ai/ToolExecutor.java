package com.stonewu.fusion.service.ai;

/**
 * 工具执行器接口
 * <p>
 * 实现此接口并注册为 Spring Bean，即可被 {@link ToolExecutorRegistry} 自动发现和调用。
 * 每个实现类自行声明工具的名称、描述和参数 schema。
 * <p>
 * 执行时通过 {@link ToolExecutionContext} 显式传递用户身份信息，
 * 不依赖 SecurityContext（ThreadLocal），兼容异步线程和微服务场景。
 * <p>
 * 示例：
 * <pre>{@code
 * @Component
 * public class ScriptQueryToolExecutor implements ToolExecutor {
 *     @Override
 *     public String getToolName() { return "script_query"; }
 *
 *     @Override
 *     public String getDisplayName() { return "查询剧本"; }
 *
 *     @Override
 *     public String getToolDescription() { return "根据条件查询剧本列表"; }
 *
 *     @Override
 *     public String getParametersSchema() {
 *         return "{\"type\":\"object\",\"properties\":{\"projectId\":{\"type\":\"integer\"}}}";
 *     }
 *
 *     @Override
 *     public String execute(String toolInput, ToolExecutionContext context) {
 *         Long userId = context.getUserId();
 *         // 解析 toolInput JSON，执行业务逻辑，返回结果 JSON
 *         return "...";
 *     }
 * }
 * }</pre>
 */
public interface ToolExecutor {

    /**
     * 工具唯一标识名称
     */
    String getToolName();

    /**
     * 工具显示名称
     */
    String getDisplayName();

    /**
     * 工具描述（告诉 AI 何时以及如何使用此工具）
     */
    String getToolDescription();

    /**
     * 工具参数的 JSON Schema
     *
     * @return JSON Schema 字符串，描述工具入参结构
     */
    String getParametersSchema();

    /**
     * 工具是否启用
     *
     * @return true 表示启用，false 表示禁用
     */
    default boolean isEnabled() {
        return true;
    }

    /** Whether the tool is guaranteed not to mutate platform state. */
    default boolean isReadOnly() {
        return false;
    }

    /** Whether separate invocations may safely execute concurrently. */
    default boolean isConcurrencySafe() {
        return false;
    }

    /**
     * 执行工具
     *
     * @param toolInput 工具入参 JSON 字符串
     * @param context   工具执行上下文（包含用户身份等信息）
     * @return 工具执行结果 JSON 字符串
     */
    String execute(String toolInput, ToolExecutionContext context);
}
