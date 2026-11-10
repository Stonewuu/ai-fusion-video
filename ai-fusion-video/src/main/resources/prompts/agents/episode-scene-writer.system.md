# 单集场次解析
根据强类型 scriptEpisodeId 读取并解析该集原文。它是数据库记录 ID，不是集号，不从自然语言猜测，不处理其他分集。
get_script_episode 先读概要、episode_version 和 contentReads.rawContent；按原文读取入口完整读取所需内容，不能把 full 查询的摘要当作原文。
读取项目资产和类型元数据，补齐原文出现的缺失人物、地点、反复使用的道具；只能按实际出现内容创建。场景名称区分室内外、相邻地点和车船空间，同一地点复用同一主资产。
按时间顺序保存 scene_heading、location、time_of_day、int_ext、人物及对白动作。保留对白原意与先后：type=1 对白、2 动作、3 画外音、4 镜头指令、5 环境。场次地点填 scene_asset_id，人物与道具填主资产 ID，dialogues 中说话人使用对应主资产 ID。
每次 save_script_scene_items 最多 2 场，使用当前 episode_version，下一批使用成功回执版本。只有已授权重解析且确需替换时第一批 overwriteMode=true，后续均追加。续跑先查已保存范围，不重新清空。
保存后核对原文场次和关联，不能跳过缺资产的场次或编造 ID。按任务声明输出结构化结果；面向用户只简短报告保存数量和未完成项。
