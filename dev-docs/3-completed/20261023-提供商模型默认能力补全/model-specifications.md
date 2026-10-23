# 规格来源与参数分层

核对日期：2026-10-23。优先使用赞助商的明确规格；未声明的尺寸、比例、时长取对应原厂资料。找不到精确型号规格时使用用户要求的通用默认模板，不称为官方规格。

## 已核对的模型

| 型号 | 配置依据与边界 |
| --- | --- |
| GPT Image 2.5 Sunburst、Flare | [OpenAI Images](https://developers.openai.com/api/reference/resources/images/methods/generate)：任意像素尺寸，16 的倍数，比例不超过 3:1，最大约 3840×2160 像素面积；提供 1K/2K/4K 映射，默认 1024×1024。 |
| Qwen Image 3.0 Pro、3.0 | [阿里云生成与编辑 API](https://www.alibabacloud.com/help/en/model-studio/qwen-image-generation-and-editing-api-reference)：512²–2048² 像素，1K/2K；没有添加超出上限的 4K。 |
| Qwen Image Edit Max、Plus | [阿里云编辑指南](https://docs.modelstudio.console.alibabacloud.com/tc/model-studio/qwen-image-edit-guide)：宽高 512–2048，预填 1K/2K。原厂图片输入能力不等于当前网关已声明对应编辑接口。 |
| Seedream 5.0 Pro、Flash | [火山 Seedream 系列](https://docs.volcengine.com/docs/ark/seedream-4-0-5-0?lang=zh)：1K/1.5K/2K，复用已核对的原厂像素表。 |
| Seedream 4.5 | [火山 Seedream 规格](https://docs.volcengine.com/docs/LakeAIService/ImagegenerationSeedreamseriesmodels?lang=zh)：2K/4K，最低像素面积 2560×1440，默认 2048×2048。 |
| Step Image Edit 2 | [阶跃官方模型文档](https://platform.stepfun.com/docs/zh/guides/models/step-image-edit-2)：采用文档公布的五组尺寸，1024×1024 默认。原厂页面注明已下线；供应商是否继续提供该 ID 仍由账号模型列表决定。 |
| Step 2X Large | 未找到该精确图片型号的公开原厂规格；按用户要求采用通用 1K/2K/4K 模板，默认 1K。没有把同品牌其他型号参数冒充此型号规格。 |
| Grok Imagine Image、Quality | [Image](https://docs.x.ai/developers/models/grok-imagine-image)、[Quality](https://docs.x.ai/developers/models/grok-imagine-image-pro)：1K/2K。保留供应商精确 ID。 |
| 即梦 Image 4.0、4.6 | [4.0](https://docs.volcengine.com/docs/JimengAI/JimengAI-ImageGeneration40-InterfaceDocumentation?lang=zh)、[4.6](https://docs.volcengine.com/docs/JimengAI/JimengAI-ImageGeneration46-InterfaceDocumentation?lang=zh)：1024²–4096² 像素，1K/2K/4K，默认 2K。 |
| Veo 3.1 Fast Preview | [Google Veo](https://ai.google.dev/gemini-api/docs/veo?hl=en)：4/6/8 秒；1080p/4K 仅 8 秒。默认 8 秒，界面与后端均限制离散时长。原厂 Preview 入口的退役不用于擅自替换供应商模型 ID。 |
| Seedance 2.0、Fast、Mini、2.5 | [2.0 系列](https://docs.volcengine.com/docs/ark/seedance-2-0?lang=zh)、[视频生成指南](https://docs.byteplus.com/en/docs/modelark/video-generation-tutorial)：2.0 系列 4–15 秒，2.5 至 30 秒；默认 5 秒。原厂支持首尾帧及参考素材，供应商通道能力单独声明。 |
| Vidu Q3 Pro、Turbo | [Vidu 模型表](https://platform.vidu.com/docs/model-map)、[Q3 API](https://platform.vidu.com/docs/api-reference/video-models/vidu-q3/image-to-video)：540p/720p/1080p，1–16 秒，默认 5 秒。 |
| 万相 3.0、Prime、2.7 | [万相 3.0](https://www.alibabacloud.com/help/en/model-studio/wan3-video-generation-guide)、[视频模型规格](https://www.alibabacloud.com/help/zh/model-studio/video-generate-edit-model)：保留供应商已实证的万相 3.0 最短 5 秒，补充相应原厂分辨率与比例选项。 |
| MiniMax H3 | [Video V2](https://platform.minimax.io/docs/api-reference/video-generation-v2-create)：768P/2K，4–15 秒；供应商 Videos 的输入字段独立限制。 |
| 可灵各型号 | [可灵官方 API](https://app.klingai.com/global/dev/document-api/apiReference/model/textToVideo)：本次页面只返回入口，正文未能读取。沿用仓库已维护的对应原厂分辨率、比例与时长，不把通用 5–15 秒覆盖到仅支持 5/10 秒的型号。 |

## 三层职责

1. 模型规格维护可选档位、比例、时长、像素映射和默认值；赞助商预设还保存独立的原厂输入能力说明。
2. 界面按模型规格生成选项和默认选择。离散时长、分辨率与时长的组合限制在界面选择和提交校验中一致执行。
3. 协议适配器编码上游字段。原厂 `resolution + ratio` 保持分离，Images/Videos 的 `size` 接口通过该模型的 `supportedSizes` 映射得到像素值，不能仅传标签或丢弃比例。

| 界面选择 | 协议输出 |
| --- | --- |
| 原厂万相：720P、16:9 | `parameters.resolution=720P`、`parameters.ratio=16:9` |
| 原厂 Seedance：720p、16:9 | `resolution=720p`、`ratio=16:9` |
| 无限星河 Videos：720p、16:9 | multipart `size=1280x720` |
| 无限星河 Videos：1080p、9:16 | multipart `size=1080x1920` |
| 图片模型：2K、16:9 | 使用该模型像素表；Qwen 为 `2688x1536`，不把所有模型的 2K 当成同一像素尺寸。 |

## 原厂能力与通道能力

[无限星河 Videos 参考](https://doc.infistar.cc/api-reference/%E8%A7%86%E9%A2%91/%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90)目前明确的提交字段为 model、prompt、seconds、size、单张 input_reference；[另一视频任务接口](https://doc.infistar.cc/api-reference/%E8%A7%86%E9%A2%91/%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90%E4%BB%BB%E5%8A%A1)没有公开首尾帧及多参考字段合同。

模型列表按用户要求只展示当前渠道已接入的能力，文案为“该渠道模型能力仅支持……”，并保留实际输入数量限制；不展示原厂能力对比或未接入能力标签。原厂资料继续作为预设维护依据，不能把通道限制解释为原厂不支持。当前适配器对尚未编码的尾帧、参考图/视频/音频明确拒绝，禁止用户手动开启能力后被静默忽略。后续取得完整供应商合同后应实现对应请求、轮询和验证，再开放通道选项；不猜测字段或自动更换协议。
