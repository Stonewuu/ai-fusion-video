# 单集上传解析
只解析当前上传分集，不改其他分集。
从 task_context 取得项目、剧本和 episode_id（剧本分集数据库 ID）。调用 get_script_episode 读取概要、episode_version、contentReads.rawContent，再按入口完整读取本集原文。上传文本是解析依据，摘要和相邻集仅用于理解上下文。
list_project_assets 与 query_asset_metadata 用于核对已有资产、按真实内容创建缺失人物、全部地点与重要道具。已有同名对象仍需核对类型和身份；不要重复创建，也不以相似地点代替缺失地点。
将原文按场次和对白动作顺序结构化，关联主资产：场景 scene_asset_id、人物 character_asset_ids、道具 prop_asset_ids；对白说话人用 character_asset_id。原文没有的剧情、人物状态和装备不自行补写。
每次 save_script_scene_items 最多 2 场，使用读取版本，后续使用回执新版本。整集重解析仅在本次已授权时替换第一批，后续追加；恢复执行不得重复清空。更新人物表前读取已有内容及版本，保留其他人物。
保存后核对场次数和资产关联，简短报告完成与缺项。

整集场次保存完成后，重新读取本集版本，用 save_script_episode 更新 synopsis（不超过 300 字，概括主要人物、事件因果和结尾状态；只写剧情，不罗列原文未交代的事项）及 parsingStatus=2；保留已保存原文，调用时不传 rawContent。尚未完成时不能标记为解析完成。跨集信息只读取梗概，不读取其他集原文。

资产枚举只使用 query_asset_metadata 返回的选项。原文没有提供的非必填属性直接不传，先创建具名的主资产；不能因为缺少时间、内外景或其他非必填属性而遗漏场景关联。创建失败后按字段定义修正参数继续，不重复提交同一个非法值。已有场次只缺关联时用 update_script_scene 补齐，续跑不重复追加已保存场次。

长原文边读边处理：每读一段，先保存已经完整识别的场次，再读取下一段；跨段场次保留所需尾段，不能先堆积整集原文再集中写入。上下文压缩时保留分集 ID、原文 contentVersion、nextOffset、已保存场次 ID、跨段未完成内容和下一步；恢复后核对当前版本。

资产查询先按名称、类型筛选分页摘要，按 hasMore、nextPage 翻页。query_asset_items 按 assetName 返回主资产候选，选定 assetId 后再查询子资产目录；需要核对完整设定时只对候选调用 get_asset 或 get_asset_item，不将 descriptionPreview 当作完整设定。
