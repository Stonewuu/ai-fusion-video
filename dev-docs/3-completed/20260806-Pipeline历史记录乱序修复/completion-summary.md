# 完成总结

## 最终范围

- 修复 AI 任务中心打开历史 Pipeline 时，子工具只有结果投影而没有调用投影导致的渲染崩溃。
- 修复参数校验失败、结果先到和父工具后到等持久化消息组合。
- 修复字符串化 JSON 因内部包含文件路径而被整体替换为 `[FILE_PATH_REDACTED]`。
- 修复普通文本和脱敏占位符点击“查看 JSON”时的异常。

## 关键实现

- 历史重建按 `toolCallId` 合并工具调用与结果；缺少调用投影时直接创建终态工具节点，后续调用到达时补充参数。
- 无调用参数的终态节点不显示“调用参数”，但仍可查看结果 JSON。
- 后端事件清洗器先识别字符串化 JSON，再递归清洗其字段，避免一个敏感路径导致整段工具结果丢失。
- 工具结果 JSON 格式化兼容结构化 JSON、普通文本、空文本和 `null`。

## 验证结果

- `pnpm exec tsc --noEmit`：通过。
- 变更文件 ESLint：通过；全量 ESLint 无错误，保留 25 条既有警告。
- 前端 Node 测试：13/13 通过。
- 后端 `AgentScopeEventMapperTests`、`AgentConfirmationExpiryCoordinatorTests`、`RunExecutionSupervisorTests`：21/21 通过。
- 用户提供的 482 条真实消息：成功重建 44 个根时间线项、格式化 175 个工具结果，其中 2 条历史脱敏占位符不再报错。
- 本地浏览器：历史 Pipeline 详情可正常打开，控制台无新增错误。

## 遗留事项

- 已经持久化为 `[FILE_PATH_REDACTED]` 的旧结果无法从已清洗历史中恢复原文；需要重新调用对应工具才能获得新结果。
