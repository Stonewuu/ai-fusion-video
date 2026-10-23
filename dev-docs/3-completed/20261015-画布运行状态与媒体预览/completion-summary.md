## 最终范围

修复明确模型错误被误判为等待确认的问题，将异常核实移到画布任务详情，并完善节点内媒体预览。用户进一步明确创作路径应允许自由组合素材与操作，分镜和先图后视频均为可选组织方式；本次没有修改现有画布的连线或工具、子 agent 定义。

## 根因与实现

- `TrackedAgentChatModel` 原先将几乎所有异常都记为 UNKNOWN_SUBMISSION。本次按 AgentScope 的 `ModelHttpException` / `HttpTransportException` 状态识别完整的 4xx / 5xx 响应，结束原请求并释放并发占用；错误原因和失败用量保留。真正断流、取消、未知传输故障不因错误文本而被误释放。
- `CanvasAgentNodeQueries` 返回当前节点对应的待核实请求编号；画布内复用共享核实对话框，使用原权限、版本校验、幂等和审计接口。任务结束后说明不再误导为“关闭后继续执行”。
- `CanvasAgentTaskProgress` 原来在 confirmation 和 selection 同时为空时，把 undefined === undefined 当成有效匹配，读取空对象造成页面异常，已修复。
- 节点媒体展示拆为纯数据提取与独立组件。输入参考和输出结果分别展示，支持图片、视频、多项结果、资产图片和镜头首尾帧。视频点击后在节点播放，可放大查看。
- 批量运行缓存保留每项最新尝试；已完成子项可以提前预览。生成执行器和专业 Agent 提供已登记的部分媒体，但不解除下游依赖，也不自动重发或改变用户素材。

## 验证

| 检查 | 结果 |
| --- | --- |
| TrackedAgentChatModelTests + TrackedAgentStreamingTests | 17 通过 |
| CanvasAgentWiringIT 选定 4 项 + CanvasRunWiringIT 选定 1 项 | 5 通过 |
| canvas-editor.test.mjs + canvas-node-runs.test.mjs | 50 通过 |
| TypeScript / 变更文件 ESLint | 通过 |
| Chrome 任务详情与原位核实 | 通过 |
| Chrome 图片与视频节点预览 | 通过；视频实际加载 320×180、播放 4 秒、无媒体错误 |
| 服务启动与代理 / Flyway validate | 通过；63 条迁移，无数据库结构改动 |

没有运行需要付费的模型生成来伪造验收结果，媒体验收使用现有测试素材。

## 当前测试画布

`雨夜灯塔 · 第一集` 的旧调用明确返回 HTTP 500（Expected canvas tool is unavailable），已通过画布内核实接口释放错误保留的占用，并留下完整依据与 1 条审计记录。对应节点正确显示 FAILED；原调用没有重发。

这张示例的首帧分支并未接入视频节点，原因是默认测试视频模型声明仅支持文生视频。此模型能力限制仍然存在，需由所选模型真实支持输入方式，不能通过连线或界面文案掩盖。

后端更新后运行于 18092，前端开发服务继续运行于 3023。刷新页面可加载本次组件和状态更新。
