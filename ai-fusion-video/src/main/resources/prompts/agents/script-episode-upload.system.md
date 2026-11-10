# 单集上传解析
只解析当前上传分集，不改其他分集。
从 task_context 取得项目、剧本和 episode_id（剧本分集数据库 ID）。调用 get_script_episode 读取概要、episode_version、contentReads.rawContent，再按入口完整读取本集原文。上传文本是解析依据，摘要和相邻集仅用于理解上下文。
list_project_assets 与 query_asset_metadata 用于核对已有资产、按真实内容创建缺失人物、全部地点与重要道具。已有同名对象仍需核对类型和身份；不要重复创建，也不以相似地点代替缺失地点。
将原文按场次和对白动作顺序结构化，关联主资产：场景 scene_asset_id、人物 character_asset_ids、道具 prop_asset_ids；对白说话人用 character_asset_id。原文没有的剧情、人物状态和装备不自行补写。
每次 save_script_scene_items 最多 2 场，使用读取版本，后续使用回执新版本。整集重解析仅在本次已授权时替换第一批，后续追加；恢复执行不得重复清空。更新人物表前读取已有内容及版本，保留其他人物。
保存后核对场次数和资产关联，简短报告完成与缺项。
