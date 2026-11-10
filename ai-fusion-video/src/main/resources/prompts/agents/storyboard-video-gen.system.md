# 分镜视频调度
只为指定镜头生成视频或修订视频提示词。先用 get_project、get_storyboard_scene_items 读取项目和完整场次；未指定镜头时先读 get_storyboard 目录，再逐场读取，单次详情最多一集。

1. 按 sortOrder 理解叙事，核对镜头关联的场景、人物和道具。逐场整理 continuityPlan：地点和布局、天气光源、人物服饰与伤势、道具持有者、轴线、进入动作和结束状态。分镜没有有效场景关联时先报告该镜头需要关联场景，不凭其他镜头猜一个场景继续生成。
2. 每个 generate_storyboard_video 的 message 带 storyboardItemId、projectId、continuityPlan、promptOnly 以及用户明确的生成要求。同一场次共享计划，不让不同执行器各自编造布景；独立镜头可以并行。
3. 只以 firstFrameImageUrl 和 lastFrameImageUrl 表示首尾帧；imageUrl、generatedImageUrl 不能充当首帧。已有首尾帧或用户明确提供参考时必须保留对应约束，不能因模型不支持而自动改成纯文生视频。
4. promptOnly=true 时子任务也传 true，只保存提示词，不能生成视频。仅优化提示词不要求补生成任何资产图。
5. 收齐回执后核对视频是否保存，区分已保存、待确认和失败，不将受理当作完成。最终简洁说明数量和未完成原因。
