# 完成摘要

## 交付范围

新增并接入 5 个明确型号：doubao-seedance-2-5-260628、MiniMax-H3、MiniMax-H3-Max、wan3.0-video、wan3.0-video-prime。分别使用火山 Content Generation V3、MiniMax Video V2 与 Wan 3.0 异步任务协议，不按模型名称猜测协议或在失败后改走其他供应商。

声明式 videoRules 统一默认值、输入组合、参考素材限制、时长/比例、原生音频、提示词及高级覆盖项。普通生成、画布、智能体经过同一套约束，运行使用冻结配置；自动时长以 -1 发送，额度按最大输出时长预留。明确失败直接返回原因，创建请求不自动重复发送。

前端增加官方供应商快捷配置、业务空间地址引导和必填校验，参数控件随模型能力变化；修复窄屏记录与高级表单布局。代理设置与工作区布局拆成独立组件，未修改 components/ui/。

## 官方依据

- Seedance 2.5：[模型说明](https://docs.volcengine.com/docs/ark/seedance-2-5?lang=zh)、[创建视频任务](https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh)。
- MiniMax H3：[Video V2 创建](https://platform.minimax.io/docs/api-reference/video-generation-v2-create)、[查询](https://platform.minimax.io/docs/api-reference/video-generation-v2-query)、[删除语义](https://platform.minimax.io/docs/api-reference/video-generation-v2-delete)。
- Wan 3.0：[API 参考](https://www.alibabacloud.com/help/zh/model-studio/wan3-video-generation-api-reference)、[生成指南](https://www.alibabacloud.com/help/zh/model-studio/wan3-video-generation-guide)。

## 验证结果

后端集中回归及定向复查最终 1,123 项通过，无跳过；前端 71 项回归、TypeScript、变更文件 ESLint、前后端构建通过。Chrome 已检查桌面/390px 窄屏、明暗主题、模型参数切换及地址校验和焦点边框。本地验收服务已更新并通过 64 条 Flyway 校验，临时模型配置已清理。

## 明确边界

未发起三家官方付费生成请求，不能据此证明具体账户与地域已开通。MiniMax 单独尾帧的官方说明矛盾暂按首尾帧成对约束；不会将其删除接口直接当作无竞争的取消接口。厂商素材库、草稿转成片等独立产品流程未纳入通用视频输入。托管素材用已有探测事实验证，外部 URL 和缺少的编码事实由供应商验证。
