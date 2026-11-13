# Vertex 接入与渠道协议筛选完成说明

## 原因与修复

当前服务账号凭据有效，Java 不显式设代理和显式使用本机 Clash 7890 均能完成 OAuth 和模型目录查询。原配置没有区域，使用 us-central1；Gemini 3.8 Flash 在该区域返回 404，在 global 成功。现已将该配置保存为 global。

原自动获取入口直接报未实现。现在按 Google 官方 v1beta1 publishers.models.list 接口分页获取，校验模型标识、去重并识别对话和图片模型；分页失败不返回半份成功结果。本次官方响应 33 项，排除未接入能力后页面显示 22 项。

鉴权类型新增 Vertex API Key，与原服务账号类型分别保存。服务账号使用项目 ID、区域和 JSON／服务器 ADC；API Key 使用 Express 全球端点，不带项目和地域路径。两种方式按显式选择执行，失败不切换凭据。API Key 请求模型目录被 Google 以 401 拒绝，页面说明应切换服务账号获取。

同时补齐 Gemini 图片模型的 generateContent 请求，保留全部参考图、传递比例和分辨率，并只保存最终图片。原 Imagen predict 策略独立保留。

秘塔和无限星河专用协议由提供商目录声明归属。API 默认协议、模型覆盖协议按渠道筛选，保存时后端再次校验；变更渠道不能绕过已有模型覆盖校验。模型/API 配置清空协议、地址或代理时，MyBatis-Plus 正确写入 NULL，避免旧值残留。

## 验证结果

- 85 项后端定向测试：分页、鉴权方式隔离、官方 SDK 文本请求格式、首段流式返回、取消、错误后不自动重发、参考图请求及专用协议限制。
- 1 项 MySQL 集成测试：升级与渠道身份保存、清空默认/覆盖协议、地址及代理；Flyway validate 通过。
- 9 项前端测试，以及 TypeScript、修改文件 ESLint；后端打包并启动成功，校验 35 条迁移。
- Chrome：服务账号获取目录并导入 Gemini 图片模型成功；Gemini 3.8 Flash 在服务账号和 API Key 下均返回 OK。
- Chrome：API Key 生成蓝色杯子，服务账号用该图作参考生成绿色杯子，均自动保存为已完成记录。没有手动绑定到项目资产。
- Chrome：普通渠道不显示赞助商专用协议，相应赞助渠道仅显示自己的专用协议；切换渠道会清除不适用的模型覆盖。检查桌面、390px 窄屏、明暗主题和焦点/错误状态后恢复原视口与浅色主题。

## 交付状态

修改保留在 v-1.3.1 分支，本地前后端已运行；未提交、未发布。Vertex 配置为服务账号 + global，已保存用户原有 API Key 供切换使用，新增一个经过验证的 Gemini 3.1 Flash Image 配置。原默认 DeepSeek 和 GPT Image 保持不变。旧秘塔配置通过页面补齐渠道身份。

API Key 无法直接查询官方目录，Imagen API Key 未接入，具体适用边界见 technical-debt.md。未修改 Google Cloud IAM、账单、API Key 限制或系统代理。

## 官方依据

- [Google Cloud API Key 接入](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/api-keys)
- [Express 请求路径](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/express-mode/api-reference)
- [Model Garden 列表接口](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1beta1/publishers.models/list)
- [Gemini 参考图生成](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/gemini-edit-images)
