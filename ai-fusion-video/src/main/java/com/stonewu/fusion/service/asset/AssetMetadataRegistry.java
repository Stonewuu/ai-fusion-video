package com.stonewu.fusion.service.asset;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 资产属性元数据注册表
 * <p>
 * 各类型资产（character/scene/prop 等）允许的 properties 字段定义。
 * 供 Controller（REST API）和 AI Tool 共享使用。
 */
public final class AssetMetadataRegistry {

        private AssetMetadataRegistry() {
        }

        // ========== 数据结构 ==========

        public record FieldDef(String key, String label, String type, boolean required,
                        List<FieldOption> options, String description) {
        }

        public record FieldOption(String value, String label) {
        }

        // ========== 元数据表 ==========

        private static final Map<String, List<FieldDef>> METADATA_MAP = new LinkedHashMap<>();

        static {
                // character（角色）
                METADATA_MAP.put("character", List.of(
                                field("gender", "性别", "select", true,
                                                List.of(opt("male", "男"), opt("female", "女"), opt("unknown", "未知")),
                                                "角色性别，必填"),
                                field("age", "年龄", "text", false, null,
                                                "写具体年龄区间，如'约二十五岁''四十至四十五岁''五十岁左右'，禁用模糊词"),
                                field("identity", "身份/职业", "text", false, null,
                                                "角色的社会身份或职业，如'高中教师''退役军人''皇帝'"),
                                field("personality", "性格特征", "text", false, null,
                                                "核心性格关键词，如'沉稳内敛''暴躁冲动'"),
                                field("appearance", "外貌特征", "text", false, null,
                                                "用中文80-150字描述可视化静态外貌，用于AI生图。"
                                                                + "人类角色须涵盖：面部五官（脸型、鼻梁、眼窝、唇厚薄、眼睛形状大小）、头发（颜色、长度、发型、发质）、"
                                                                + "体型（身高感、肩宽、腰线）、皮肤（仅质感和独特标记如雀斑/疤痕/纹身，禁写肤色）、"
                                                                + "鞋子（必填，款式颜色材质）、配饰（耳钉/项链/手表等）。"
                                                                + "非人类角色以名称/物种名开头自由描述核心特征。"
                                                                + "禁忌：禁写身体部位颜色（肤色/唇色/眼色/脸色，头发和服装颜色可写）、"
                                                                + "禁写表情姿态动作情绪、禁写背景环境道具光影画风、禁写抽象气质、禁用不确定词（或/可能/也许/大概）。"
                                                                + "服装发型配饰须符合故事年代设定，原文有外貌描述时以原文为最优先（颜色禁忌仍需遵守）"),
                                field("clothing", "服装描述", "text", false, null,
                                                "款式、材质、配色、细节，如'黑色机车皮夹克搭配破洞牛仔裤，金属拉链装饰'。须符合故事年代设定"),
                                field("relationship", "人物关系", "text", false, null,
                                                "与其他角色的关系，如'张三的妻子''王五的上司'")));

                // scene（场景）
                METADATA_MAP.put("scene", List.of(
                                field("locationType", "场景类型", "select", false,
                                                List.of(opt("interior", "内景"), opt("exterior", "外景"),
                                                                opt("mixed", "内外景")),
                                                "该场景是室内、室外还是内外混合"),
                                field("atmosphere", "氛围/基调", "text", false, null,
                                                "场景整体氛围关键词，如'阴森压抑''温馨明亮''荒凉萧瑟'"),
                                field("timeOfDay", "时间段", "select", false,
                                                List.of(opt("day", "日"), opt("night", "夜"),
                                                                opt("dawn", "晨"), opt("dusk", "黄昏")),
                                                "场景默认时间段"),
                                field("description", "场景描述", "text", false, null,
                                                "场景的空间布局、陈设、建筑风格等可视化特征描述，用于AI生图。须符合故事年代设定")));

                // prop（道具）
                METADATA_MAP.put("prop", List.of(
                                field("category", "道具类别", "select", false,
                                                List.of(opt("weapon", "武器"), opt("food", "食物"),
                                                                opt("tool", "工具"), opt("decoration", "装饰品"),
                                                                opt("document", "文书"), opt("other", "其他")),
                                                "道具的功能分类"),
                                field("material", "材质", "text", false, null,
                                                "道具的主要材质，如'青铜''玉石''不锈钢'"),
                                field("significance", "剧情意义", "text", false, null,
                                                "该道具在剧情中的作用和象征意义")));

                // ========== 以下类型暂时屏蔽 ==========

                // vehicle（载具）
                // METADATA_MAP.put("vehicle", List.of(
                // field("vehicleType", "载具类型", "select", false,
                // List.of(opt("car", "汽车"), opt("horse", "马匹"),
                // opt("boat", "船"), opt("aircraft", "飞行器"),
                // opt("other", "其他")),
                // "载具的大类"),
                // field("description", "载具描述", "text", false, null,
                // "载具的外观、品牌型号（如适用）、颜色、细节特征描述")
                // ));

                // building（建筑）
                // METADATA_MAP.put("building", List.of(
                // field("buildingType", "建筑类型", "select", false,
                // List.of(opt("residence", "住宅"), opt("commercial", "商业"),
                // opt("temple", "寺庙/宗教"), opt("palace", "宫殿"),
                // opt("fortress", "要塞"), opt("other", "其他")),
                // "建筑的功能分类"),
                // field("era", "时代风格", "text", false, null,
                // "建筑的时代和建筑风格，如'唐代木构''哥特式''现代简约'"),
                // field("description", "建筑描述", "text", false, null,
                // "建筑的外观、结构、规模、装饰等可视化特征描述")
                // ));

                // costume（服装）
                // METADATA_MAP.put("costume", List.of(
                // field("style", "服装风格", "text", false, null,
                // "服装的整体风格，如'中式旗袍''西式礼服''赛博朋克'"),
                // field("color", "主色调", "text", false, null,
                // "服装的主要配色，如'深红配金''黑白撞色'"),
                // field("occasion", "适用场合", "text", false, null,
                // "该服装适用的场合，如'婚礼''战斗''日常'")
                // ));

                // effect（特效）
                // METADATA_MAP.put("effect", List.of(
                // field("effectType", "特效类型", "select", false,
                // List.of(opt("fire", "火焰"), opt("water", "水"),
                // opt("lightning", "闪电"), opt("magic", "魔法"),
                // opt("explosion", "爆炸"), opt("weather", "天气"),
                // opt("other", "其他")),
                // "特效的大类"),
                // field("description", "特效描述", "text", false, null,
                // "特效的视觉表现描述，如颜色、形态、运动方式等")
                // ));
        }

        // ========== 公共方法 ==========

        public static List<FieldDef> getFields(String assetType) {
                return METADATA_MAP.get(assetType);
        }

        public static Set<String> getSupportedTypes() {
                return METADATA_MAP.keySet();
        }

        // ========== 辅助方法 ==========

        private static FieldDef field(String key, String label, String type, boolean required,
                        List<FieldOption> options, String description) {
                return new FieldDef(key, label, type, required, options, description);
        }

        private static FieldOption opt(String value, String label) {
                return new FieldOption(value, label);
        }
}
