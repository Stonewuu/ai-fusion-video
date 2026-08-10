# ComfyUI 对接开发实施手册

## 1. 文档目的

本文档是 v1.1.0 ComfyUI 对接的长期实施基线，记录范围、架构、协议、数据结构、用户流程、实现顺序和验收标准。实现发生调整时必须先更新本文档，再更新代码和进度文档，避免上下文压缩后丢失决策。

## 2. 目标与边界

### 2.1 目标

- 管理员可以配置一个 ComfyUI Native API 实例并检测连接。
- 管理员可以导入 ComfyUI API-format 工作流，保存 UI-format 源文件（可选），配置输入/输出绑定，验证、试运行、发布和创建新版本。
- 图片、视频模型可以绑定已发布工作流，并通过现有任务队列执行。
- 平台负责上传输入素材、提交工作流、轮询任务、取消任务、下载结果并转存到平台存储。
- 普通用户继续使用现有图片/视频生成表单，不接触 ComfyUI 节点 JSON。
- 所有远端协议行为以 ComfyUI 官方文档和 v0.30.0 源码契约为准。

### 2.2 本次明确不做

- 不允许普通用户提交任意工作流 JSON。
- 不自动安装 checkpoint、LoRA、自定义节点或 Python 依赖。
- 不通过运行时探测在多个协议之间静默 fallback。
- 第一版不接 WebSocket 实时采样进度；任务状态以 HTTP 轮询为准。
- 第一版不接 Comfy Cloud 和 Comfy API v2；为后续适配保留客户端边界。

### 2.3 本地环境限制

当前开发环境未部署 ComfyUI，因此本阶段完成：代码实现、静态校验、单元测试、HTTP 契约测试和构建验证。真实 GPU 工作流联调、custom node 兼容性验证和输出媒体实测，待用户提供 ComfyUI 环境后执行，必须记录在技术债务文档中，不得声称已完成运行态验收。

## 3. 官方协议基线

Native API 基线：ComfyUI v0.30.0。

| 能力 | 方法与路径 | 平台用途 |
| --- | --- | --- |
| 服务信息 | `GET /system_stats` | 连接检测与设备信息 |
| 服务能力 | `GET /features` | 能力检测 |
| 节点定义 | `GET /object_info/{node_class}` | 导入校验 |
| 模型目录 | `GET /models/{folder}` | checkpoint/LoRA 等依赖诊断 |
| 上传图片 | `POST /upload/image` | `LoadImage` 输入 |
| 提交任务 | `POST /prompt` | 提交 API-format 工作流 |
| 查询任务 | `GET /api/jobs/{prompt_id}` | 权威轮询入口 |
| 取消任务 | `POST /api/jobs/{prompt_id}/cancel` | 用户取消 |
| 下载输出 | `GET /view` | 按 filename/subfolder/type 获取媒体 |

任务状态映射：

| ComfyUI | 平台 |
| --- | --- |
| `pending` | 处理中（远端排队） |
| `in_progress` | 处理中 |
| `completed` | 解析并持久化结果后完成 |
| `failed` | 失败 |
| `cancelled` | 失败/取消（以平台任务状态能力为准） |

提交请求由平台预生成 canonical UUID 并传入 `prompt_id`。当提交请求结果不确定时，先按该 UUID 查询任务，不直接重复提交。

v0.30.0 精确响应契约：`/system_stats` 的版本字段为 `system.comfyui_version`；`/object_info/{node_class}`
返回以 class_type 为键的对象；`/upload/image` 返回 `name`、`subfolder`、`type`；Jobs 完成态的 `outputs`
按节点 ID 分组，节点输出中的文件描述包含 `filename`、`subfolder`、`type`，再以这三项请求 `/view`。平台不猜测
其他字段位置或旧版任务接口。

## 4. 顶层架构

### 4.1 核心对象

- `ApiConfig`：ComfyUI 实例地址、Bearer Token、代理配置。
- `ComfyUiWorkflow`：一个业务工作流的稳定身份、类型与当前发布版本。
- `ComfyUiWorkflowVersion`：不可变执行版本，保存 API JSON、可选 UI JSON、绑定、依赖和校验结果。
- `AiModel`：用户可选择的业务模型，关联 ComfyUI 工作流。
- `ImageTask` / `VideoTask`：提交时固定工作流版本，防止排队期间发布新版本导致执行漂移。
- `ComfyUiClient`：Native API HTTP 边界。
- `ComfyUiWorkflowRenderer`：深拷贝模板并按绑定注入业务参数。
- `ComfyUiImageStrategy` / `ComfyUiVideoStrategy`：接入现有生成策略路由。

### 4.2 分层约束

1. Controller 只处理鉴权、参数校验和 VO 转换。
2. Workflow Service 负责版本、发布和业务一致性。
3. Client 只负责 HTTP、认证、代理、DTO 和协议错误。
4. Renderer 不访问网络和数据库，保持纯函数以便完整单测。
5. Strategy 负责任务到工作流参数的映射编排，不复制 HTTP 细节。
6. 输出必须由 ComfyUI Client 携带认证流式下载到受控临时文件，再调用 `MediaStorageService.storeFile`；不把受保护的 `/view` URL 交给无认证通用下载器。

## 5. 数据设计

### 5.1 `afv_comfyui_workflow`

- `id`
- `api_config_id`
- `name`
- `code`
- `model_type`：2 图片、3 视频
- `description`
- `active_version_id`
- `status`：0 禁用、1 启用
- `deleted`、`deleted_id`
- `create_time`、`update_time`

同一个 API 配置下 `code + deleted_id` 唯一。
`api_config_id` 是工作流创建后的不可变归属；需要迁移到其他 ComfyUI 供应商时必须在目标供应商下新建工作流并重新校验、试运行和发布。

### 5.2 `afv_comfyui_workflow_version`

- `id`
- `workflow_id`
- `version_no`
- `ui_workflow_json`：LONGTEXT，可空
- `api_workflow_json`：LONGTEXT，必填
- `input_bindings_json`：LONGTEXT，结构化绑定
- `output_bindings_json`：LONGTEXT，结构化输出选择
- `required_nodes_json`：LONGTEXT，导入时提取的 class_type 清单
- `workflow_hash`：SHA-256
- `validation_status`：0 未验证、1 通过、2 失败
- `validation_message`：LONGTEXT
- `test_status`：0 未试运行、1 成功、2 失败
- `test_message`：LONGTEXT
- `last_test_time`：最后一次试运行时间
- `published`：是否曾发布
- `create_time`、`update_time`

已发布版本不可原地修改；编辑操作创建下一版本。发布时更新主表 `active_version_id`。回滚通过重新指向旧的已验证版本完成。

### 5.3 现有表扩展

- `afv_ai_model.comfyui_workflow_id`：仅 ComfyUI 图片/视频模型使用。
- `afv_image_task.workflow_version_id`：任务提交时固化。
- `afv_video_task.workflow_version_id`：任务提交时固化。

当前 `main` 的 `V1.1.0.0.0` 至 `V1.1.0.0.8` 属于产品 `1.0.0`。本功能属于产品 `1.1.0`，因此使用
`V1.1.1.0.0__comfyui_workflow_support.sql`，不得改写任何既有迁移或基线。文件名中的产品版本段为 `1.1.0`，最后一段是该产品版本内从 `0` 开始的迁移序号。

## 6. 工作流与绑定格式

### 6.1 API-format 判定

- 顶层必须为对象。
- 禁止顶层出现 UI-format 的 `nodes` 与 `links`。
- 每个节点值必须含非空 `class_type` 和对象类型 `inputs`。
- 节点 ID 统一按字符串保存。
- 导入大小、节点数、绑定数设上限。
- 检测疑似密钥字段，禁止把密钥写入版本 JSON。

### 6.2 输入绑定

统一业务字段：

- 公共：`prompt`、`negativePrompt`、`seed`、`count`。
- 图片：`width`、`height`、`referenceImages`。
- 视频：`width`、`height`、`duration`、`fps`、`firstFrame`、`lastFrame`、`referenceImages`、`referenceVideos`、`referenceAudios`、`generateAudio`。

单个字段可以绑定多个目标：

```json
{
  "prompt": [
    { "nodeId": "6", "inputName": "text", "valueType": "string" }
  ],
  "referenceImages": [
    { "nodeId": "12", "inputName": "image", "valueType": "uploaded_image", "index": 0 }
  ]
}
```

绑定必须显式，不依赖节点名称或模型代码猜测；导入 UI 可以提供建议，但管理员必须确认。

### 6.3 输出绑定

```json
[
  { "nodeId": "9", "mediaType": "image", "role": "primary" },
  { "nodeId": "31", "mediaType": "video", "role": "primary" },
  { "nodeId": "35", "mediaType": "image", "role": "cover" }
]
```

只处理显式绑定节点。中间预览、遮罩、调试输出不得自动暴露。

## 7. 管理员用户流程

1. 在 ComfyUI 中制作并完整运行工作流。
2. 保存 UI-format JSON 用于以后编辑。
3. 使用 `File -> Export (API)` 导出 API-format JSON。
4. 平台模型管理中创建 ComfyUI API 配置并测试连接。
5. 在该 ComfyUI 供应商卡片中进入工作流管理；新建工作流自动绑定当前供应商，再导入两个 JSON 并选择图片/视频类型。
6. 平台解析节点并向目标实例校验 class_type 与依赖。
7. 管理员映射业务输入与输出节点，配置能力范围。
8. 平台先做目标实例在线校验，再执行真实测试任务；两项均成功后才允许发布。
9. 发布版本并将工作流绑定到 `AiModel`。
10. 普通用户在现有生成页面选择模型并提交任务；排队任务可直接取消，运行中的 ComfyUI 任务会同时取消远端 prompt。

## 8. 运行时流程

1. 提交接口校验模型能力并写入本地 Task/Item。
2. 对 ComfyUI 模型解析当前发布版本并写入 `workflow_version_id`。
3. Consumer 获取固定版本，Renderer 深拷贝 API JSON并注入参数。
4. 输入图片由平台读取为字节并上传 `/upload/image`，返回文件名注入工作流。
5. Client 生成 UUID、先持久化到 Item，再 `POST /prompt`。
6. Strategy 轮询 `/api/jobs/{id}`，支持取消和超时。
7. 完成后仅按输出绑定解析文件描述。
8. Client 通过 `/view` 流式下载到受控临时文件，校验大小和数量。
9. `MediaStorageService.storeFile` 保存结果并更新 Item，临时文件随后删除。
10. 所有结果处理成功后更新 Task 为完成；远端或持久化失败均记录明确错误。

## 9. 前端设计

- `PLATFORM_OPTIONS` 增加 ComfyUI。
- API 配置 Dialog 对 ComfyUI 显示实例地址、Bearer Token、代理和连接测试，不显示远程模型同步入口。
- 模型 Dialog 在选择 ComfyUI 图片/视频模型时显示工作流选择器，模型协议固定为 `comfyui`。
- 选择工作流后按已发布版本的绑定同步图片参考能力；传输格式明确写为 `url` 与 `data_uri`，视频/音频文件上传能力不在第一版宣告。
- 工作流入口位于对应的 ComfyUI 供应商卡片，目录与新建操作固定按 `apiConfigId` 隔离，不提供脱离供应商的全局入口。
- 新增路由专用组件，承载工作流列表、导入向导、绑定编辑器、版本记录和测试结果，避免继续扩大现有页面文件。
- 导入向导控制在六步：基本信息、导入、依赖检查、输入绑定、输出绑定、测试发布。
- 浮层、按钮、Select、滚动严格遵守项目 `AGENTS.md` 设计系统。

## 10. 安全与可靠性

- 管理接口仅管理员可用。
- ComfyUI 应部署在私网或带认证的反向代理后，禁止裸露 8188。
- Bearer Token 不写日志、不写工作流 JSON。
- workflow JSON 仅作为数据解析，不允许执行表达式或脚本。
- 自定义节点安装仍由 ComfyUI 运维负责；平台只检测，不自动安装。
- 文件下载只使用 ComfyUI 返回的 filename/subfolder/type，通过官方 `/view` 获取，限制 MIME、单文件大小和总输出数量。
- 发布与回滚事务化；任务引用不可变版本。

## 11. 实施阶段与验收

### 阶段 A：基线与文档

- 分支从 `main` 创建。
- 三份文档落库。

### 阶段 B：工作流领域

- 迁移、实体、Mapper、Service、Controller、VO。
- CRUD、版本、校验状态、发布、回滚、模型绑定。

### 阶段 C：协议与渲染

- Native Client、DTO、认证/代理、导入解析、绑定渲染、输入上传、输出下载。

### 阶段 D：生成接入

- 图片/视频 Strategy、路由、版本固化、任务轮询、取消、结果持久化。

### 阶段 E：前端闭环

- 连接检测、工作流导入绑定、版本发布、模型关联。

### 阶段 F：验证与交付

- 后端单测、HTTP 契约测试、Flyway 命名测试、后端构建、前端 lint/typecheck/build。
- 记录真实 ComfyUI 联调缺口。
- 提交全部变更。

## 12. 当前验证记录

- `FlywayMigrationNamingTests`：2/2 通过，确认产品 `1.1.0` 首条迁移为 `V1.1.1.0.0`。
- ComfyUI 文档、Renderer、Native Client、输入/输出资源、领域服务、模型绑定、图片/视频 Strategy、版本固化与取消
  路径，加上 Flyway 命名测试：选定后端验证集 46/46 通过。
- Native Client 已按官方 v0.30.0 `server.py` 与 `comfy_execution/jobs.py` 终审；连接版本字段不做 fallback，未知
  `Content-Length` 的 JSON、输入图片和输出下载也分别在缓冲/写入前执行 16MB、20MB、1GB 硬上限。
- 前端：`pnpm exec tsc --noEmit`、`pnpm lint`（0 error）和 `pnpm build` 通过。
- 后端全量 `mvn test` 已运行；除 3 个依赖本地 MySQL 的 Spring 上下文测试类因 `Connection refused` 失败外，其余报告无失败。真实 Flyway validate 与 ComfyUI/GPU 联调仍受外部环境限制。

## 13. 官方资料

- https://docs.comfy.org/development/comfyui-server/comms_routes
- https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/script_examples/basic_api_example.py
- https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/script_examples/websockets_api_example.py
- https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/server.py
- https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/comfy_execution/jobs.py
