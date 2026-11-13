# 模型联网匹配、Gemini 预设与全局文案规范

## 最终功能

模型编辑新增“联网匹配”，从 Models.dev 读取公开目录并缓存 1 小时。用户可按名称、完整模型 ID 或接近的拼写搜索，查看渠道和能力，选中后应用，再保存到当前模型。目录可以手动刷新，失败可重试，未知参数显示“—”，来源和日期放在详情中。

匹配只更新用户明确应用的能力字段，不更换模型调用 ID、API 配置、请求协议和并发设置。已有自定义配置继续保留，匹配来源写入 config.catalogMatch。搜索只在本地读取的目录上执行，不向 Models.dev 发送搜索词或 API 密钥。

Google 官方预设现有 19 项：14 项文本、5 项图片，分别保存完整模型代码、官方来源和核对日期。官方图片预设采用 google: 前缀，与 Flow 的同名预设分开；Flow 原有 4 项图片预设的能力数据未改，用户仍可修改。逐项对应关系及未接入类型记录在 model-correspondence.md。

远程获取按渠道和完整模型 ID 选择预设；正式版和预览版不再被 baseModelId 合并。Gemini Omni 不再误归为对话模型。官方图片参数补齐比例、尺寸档位和参考图上限，调用时保留用户选择的比例和档位，支持 512 及超宽／超高比例。新增 Gemini Developer API 图片协议，与 Vertex 使用不同地址和凭据。

Gemini 3／4 不再自动携带旧版 thinkingBudget，思考等级统一转换成 Google 的枚举值。Gemini 2.5 保留预算方式，Gemini 4 记录共享上下文以及单次输出上限，未把长解码能力当成单次输出能力。

## 全局规则

用户级 AGENTS.md 增加“全局产品文案规范”，覆盖所有项目、模块、功能以及所有用户可见状态；本项目规则同步。要求使用简洁的操作和状态文案，开发背景和防御性说明不进入产品界面，必要的费用、权限和不可撤销后果仍须准确说明。当前修改涉及的模型表单文案已按此规范整理。

## 验证

- 后端 47 项定向用例通过，覆盖官方与渠道预设隔离、目录分类、完整版本 ID、模糊搜索排序、字段缺失语义、图片尺寸组合、Gemini 思考等级的出站 HTTP 请求及现有模型能力规则。
- 前端 5 项用例验证按渠道筛选协议、匹配时保留 Flow 和自定义配置、保留调用标识、未知值与明确不支持的区别，以及模态传输设置；TypeScript 与修改文件 ESLint 通过。
- Chrome 验证公开目录联网获取、拼写近似搜索、预览、应用、保存和重新读取。模型 18 现为 gemini-3.8-flash 预设，模型 19 现为 google:gemini-3.1-flash-image 预设；调用 ID、渠道及原默认模型未变。
- Chrome 检查桌面、390px 窄屏、明暗主题、焦点、加载、空结果、禁用、请求失败和恢复。临时网络阻断已撤销，视口与浅色主题已恢复。
- 使用现有 Vertex 服务账号生成 512 档、1:4 图片成功，返回 256×1024 并自动保存。本轮没有逐一付费调用全部预设，也没有进行 Gemini Developer API 密钥的外部生成测试。
- 后端打包成功、本地服务已重启，启动时 Flyway 校验 35 条迁移；本轮没有新增数据库迁移。

## 状态与边界

代码位于 v-1.3.1 工作分支，尚未提交或发布。实时语音、语音合成、嵌入、转写、Omni 原生视频和托管研究智能体属于独立调用能力，未伪装成可用对话预设；官方尺寸表的一处不一致及兼容范围记录在 technical-debt.md。

Models.dev 接入依据：[公开 API 与字段说明](https://github.com/anomalyco/models.dev#api)。Google 依据：[模型目录](https://aistudio.google.com/docs/models)、[Vertex 模型目录](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models)、[思考配置](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking)、[图片参数](https://aistudio.google.com/docs/generate-content/image-generation)。
