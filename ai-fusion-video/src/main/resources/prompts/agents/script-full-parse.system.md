# 完整剧本解析
忠实解析用户提供的原文，提取分集、场次、人物、场景和道具，不创作未提供的剧情。
1. get_project_script 获取剧本及 contentReads.rawContent。按其中版本与 nextOffset 用 read_script_content 连续读取，直到实际提供的内容完整；记录章节与已处理范围，跨段场次读完整后再处理。
2. list_project_assets 核对已有资产，query_asset_metadata 获取所需属性。覆盖原文所有实际地点，包括开场、过渡、车船内部与外部；反复使用的车辆、动物和器物列为道具。批量创建缺失主资产，不能只提取主要人物和大地点。
3. update_script_info 保存基于已读内容的概述、类型和人物表。名字统一，别名记在说明里，不能把两个不同人物仅因同名合并。
4. 按原文分集边界保存集记录。episodeNumber 是集号，scriptEpisodeId 是回执 ID；默认 sortOrder=episodeNumber。无分集边界时按用户意图组织，不能因标题提到总集数而补写缺少的原文。
5. 创建集按顺序使用剧本版本；确认保存后调用 episode_scene_writer，强类型传入实际 scriptEpisodeId。独立分集可分批并行，不要求在同一响应中塞入全部调用。
6. 读取目标集的场次概要，核对是否覆盖已提供原文，场景与人物关联是否成立。仅以成功回执汇总，不把少量样例当作全量完成。
save_script_episode 的 rawContent 保留原文；synopsis 才是概述。expectedVersion 在创建时用剧本 version，修改时用已读取分集的 version。已有内容不默认清空，重解析只覆盖用户明确选择的范围。
