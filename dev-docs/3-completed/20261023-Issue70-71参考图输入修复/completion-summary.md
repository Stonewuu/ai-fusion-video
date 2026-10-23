# 完成总结

## 核心结论

- [Issue #70](https://github.com/Stonewuu/ai-fusion-video/issues/70)：已确认本站绝对 URL 转为 Data URI 时，原代码没有识别其本地资源身份，会发起不必要的网络下载。但这不足以单独归因用户现场，仍须区分前端 API 地址、代理转发、后端资源公网地址以及模型输入格式。当前代码的下载异常会继续传到执行器、失败项与任务聚合状态，未复现“下载失败仍无参考图生成成功”。报告中的设置返回后显示问题在当前代码未发现，保存值优先于自动探测值。
- [Issue #71](https://github.com/Stonewuu/ai-fusion-video/issues/71)：核心是模型声明数量、工作流绑定和实际消费不一致，而非全局只能传两张。仓库没有该两槽默认工作流；已有按索引绑定与模型选择后的数量推导。真实缺陷是多余或未绑定素材可以在渲染时被忽略。四张图片完整绑定的原有行为正常。
- [ComfyUI 官方节点文档](https://github.com/Comfy-Org/docs/blob/main/built-in-nodes/TextEncodeQwenImage21.mdx)确认 TextEncodeQwenImage21 最多 16 张；节点上限不等于每份工作流的实际容量，也不能代替显存和质量验证。

## 已实施修改

1. 仅对与 resource_base_url 的协议、主机、端口和路径前缀严格一致的本站媒体 / 画风地址还原本地资源路径，直接读取。保留外部地址下载、路径边界和托管资源授权逻辑。
2. 参考图处理失败记录包含模型与图片序号的警告并继续抛错；日志不记录可能包含签名或凭据的原始 URL。保持现有任务失败原因回传，不自动丢弃素材继续生成。
3. ComfyUI 上传前检查媒体绑定覆盖，拒绝数量超出槽位、索引缺口、完全未绑定、单文件槽位接收多图和缺少必需索引。明确接收整份列表的节点继续支持 string_list。
4. 系统设置保留简短用途说明，工作流编辑保留通用多图绑定指引；部署和排查信息不进入常规产品文案。修复两处说明区域的窄屏排列。
5. 已将通用文案规则写入项目 AGENTS.md：常驻文案只承载当前操作与决策需要的信息；排查过程、修复理由、内部实现和个别案例进入开发记录或问题回复。

## 四图配置说明

模型上限应与实际工作流一致。若工作流已包含四个图片加载节点并接入生成链路，可将 referenceImages 的 0、1、2、3 分别绑定到各 LoadImage.image，并把该模型 maxReferenceImages 配置为 4。更多图片按同一规则扩展，不在执行器写死四张。

```json
{
  "referenceImages": [
    {"nodeId":"load_1","inputName":"image","valueType":"uploaded_image","index":0},
    {"nodeId":"load_2","inputName":"image","valueType":"uploaded_image","index":1},
    {"nodeId":"load_3","inputName":"image","valueType":"uploaded_image","index":2},
    {"nodeId":"load_4","inputName":"image","valueType":"uploaded_image","index":3}
  ]
}
```

示例仅展示绑定格式，节点 ID 需对应实际 API-format 工作流；增加绑定不会自动增加生成链路中的图片插槽。

## 验证

- 修复前：两项本站绝对 URL 回归出现连接 localhost 失败；三项缺失绑定回归确认原执行器没有拒绝输入。
- 修复后：相关 62 项后端测试通过，涵盖本地 / 外部资源分流、URL 编码及路径边界、下载失败、四图顺序映射、完整列表传递和已有图片 / 视频策略。
- 前端 ESLint、TypeScript 检查通过，Chrome 真实组件隔离预览检查桌面 / 窄屏、明暗主题、首尾与中部焦点、非法 URL 和禁用状态。未修改生产路由。
- 未调用收费模型或真实 GPU 出图；未修改用户模型记录、工作流、Compose 网络配置和数据库迁移。未向 GitHub 发布评论。
- 最新代码已加载到本地后端，现有 Docker 数据库连接正常，65 项 Flyway 迁移校验通过，HTTP 检查返回 200。

## Compose 与代理补充核验

- 已核对当前工作区和 GitHub main 的 Compose / Nginx 文件：默认 PUBLIC_API_URL 为空，浏览器采用同源路径；Nginx 将 /api/（包括画风图）及 /media/ 原样转给 backend:18080。
- PUBLIC_API_URL 是浏览器运行时 API 根地址；resource_base_url 是存储在 afv_system_config 中的系统资源根地址，主要用于后端构造外部资源 URL；backend:18080 只用于容器内部转发。三者用途不能混用。
- 使用现有 nginx:1.30.5-alpine 与 Python 桩服务建立隔离 Docker 网络，验证服务选择、路径和查询参数、Host、HTTP / HTTPS 转发，以及运行时配置禁止缓存。未连接用户应用或数据库容器。
- 确认 /media/ 原先把外层 X-Forwarded-Proto=https 覆盖成 http，而 /api/ 等路径保留 https；已统一为现有 fusion_forwarded_proto 映射。修复后 10 种转发组合全部通过，nginx -t 通过；不能据此断言该独立问题就是报告的参考图失效原因。
- Compose 默认 / 显式 API 地址 / 分域三种配置解析通过；现有两个前端 runtime-api 回归通过。修正 README 和 .env.example 中“PUBLIC_API_URL 只在分域部署生效”的不准确表述，说明统一入口必须留空。
- 模型配置允许 data_uri/base64 时，内网 URL 才会进入该参考图下载转换路径；仅 URL 模式会先拒绝无法供云端访问的地址。配置声明不等于上游实际能力，仍需核对实际模型、协议及已保存配置。
- 未取得报告者实际生效的 Compose、外层反代、镜像 digest 或模型配置，用户现场的最终根因仍不能完全确定。临时容器与网络已在验证后清理。
