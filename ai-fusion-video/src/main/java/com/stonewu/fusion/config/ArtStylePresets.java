package com.stonewu.fusion.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

/**
 * 预设画风注册表
 * <p>
 * 静态定义所有预设画风配置，包括名称、描述、英文提示词和参考图路径。
 * 预设参考图存放在 resources/static/art-styles/ 目录下，并统一通过 /api/art-styles/** 对外暴露。
 */
public class ArtStylePresets {

    private static final Map<String, ArtStylePreset> PRESETS = new LinkedHashMap<>();

    static {
        register(ArtStylePreset.builder()
                .key("cartoon_3d")
                .name("卡通3D")
                .description("高品质3D卡通动画风格，精细3D渲染，夸张且富有表现力的角色比例，柔和全局光照与体积光，次表面散射通透皮肤材质，细腻毛发与织物纹理，鲜艳明快温馨色彩，电影级构图与柔和景深")
                .imagePrompt("高品质3D卡通动画风格，精细的3D渲染，夸张且富有表现力的角色比例，柔和的全局光照（Global Illumination），体积光，次表面散射材质（Subsurface Scattering，使皮肤具有通透感），细腻的毛发与织物纹理，鲜艳明快且温馨的色彩搭配，电影级构图，柔和的景深效果，Octane Render级别的极致画质，极具亲和力的生动设计")
                .referenceImagePath("/api/art-styles/cartoon_3d.jpg")
                .build());

        register(ArtStylePreset.builder()
                .key("cg")
                .name("CG动画")
                .description("顶级次世代CG动画风格，超高精度3D模型，基于物理的渲染（PBR），史诗级宏大场景，电影级光线追踪，复杂粒子特效，细腻皮肤微观纹理与硬表面反射，强烈明暗对比与戏剧性打光，8K超高分辨率")
                .imagePrompt("顶级次世代CG动画风格，超高精度3D模型，基于物理的渲染（PBR），史诗级宏大场景，电影级光线追踪（Ray Tracing），复杂的粒子特效（魔法、尘埃、火花），细腻的皮肤微观纹理与硬表面反射，强烈的明暗对比与戏剧性打光，环境光遮蔽（AO），8K超高分辨率，极具视觉冲击力的特效，极致的写实细节与史诗感")
                .referenceImagePath("/api/art-styles/cg.jpg")
                .build());

        register(ArtStylePreset.builder()
                .key("realistic")
                .name("写实")
                .description("好莱坞电影级写实摄影，超写实主义，35mm电影镜头，f/1.8大光圈极浅景深，专业电影级打光，变形镜头眩光，高级电影胶片质感，专业电影级调色，8K分辨率")
                .imagePrompt("好莱坞电影级写实摄影，超写实主义，实景真人画风，使用35mm电影镜头拍摄，f/1.8大光圈极浅景深，专业的电影级打光（伦勃朗光、边缘背光），变形镜头眩光（Anamorphic lens flare），高级电影胶片质感与微小的胶片颗粒，专业的电影级调色（青橙色调 Teal and Orange），极度清晰的面部毛孔与微距质感，8K分辨率，极其逼真的真实世界光影")
                .referenceImagePath("/api/art-styles/realistic.jpg")
                .build());

        register(ArtStylePreset.builder()
                .key("anime_jp")
                .name("日漫")
                .description("高品质2D日式动画风格，赛璐璐涂装，清晰细腻轮廓线稿，平涂上色与层次分明硬边缘阴影，高饱和纯净动漫色彩，唯美光影氛围与空气感，精致手绘背景")
                .imagePrompt("高品质2D日式动画风格，赛璐璐涂装（Cel Shading），清晰细腻的轮廓线稿，平涂上色与层次分明的硬边缘阴影，高饱和且纯净的动漫色彩，富有张力的角色神态与动作，唯美的光影氛围与空气感，丁达尔效应，精致细腻的手绘背景（2D background art），高质量动画截图质感，带有经典日式动画的光晕与唯美滤镜")
                .referenceImagePath("/api/art-styles/anime_jp.jpg")
                .build());

        register(ArtStylePreset.builder()
                .key("anime_cn")
                .name("国漫")
                .description("新国风高级动画风格，融合中国传统水墨画与现代数字插画技法，写意与工笔结合，流畅飘逸线条与灵动水墨晕染，东方传统色彩美学，武侠与东方奇幻氛围，气韵生动")
                .imagePrompt("新国风高级动画风格，融合中国传统水墨画与现代数字插画技法，写意与工笔结合，流畅飘逸的线条与灵动的水墨晕染效果（Ink wash painting），使用传统的东方色彩美学（朱砂、石青、藤黄），武侠与东方奇幻氛围，强烈的气韵生动感，带有粗犷的毛笔笔触质感，神秘且空灵的古典光影，2D与3D结合的高级手绘质感")
                .referenceImagePath("/api/art-styles/anime_cn.jpg")
                .build());

        register(ArtStylePreset.builder()
                .key("comic_us")
                .name("美漫")
                .description("现代美式漫画与前卫动作动画风格，极强视觉张力，粗犷动感黑色轮廓线条，极具夸张爆发力的透视构图，高对比度波普色彩，半调网点效果，色散偏移与故障艺术视觉效果")
                .imagePrompt("现代美式漫画与前卫动作动画风格，极强的视觉张力，粗犷且动感的黑色轮廓线条，极具夸张与爆发力的透视构图，高对比度的波普（Pop art）色彩，带有印刷半调网点（Halftone pattern）效果，色散偏移（Chromatic aberration），带有毛刺与故障艺术（Glitch art）的视觉残影与伪影效果，极度硬核且风格化的动态模糊，浓重的墨迹阴影")
                .referenceImagePath("/api/art-styles/comic_us.jpg")
                .build());
    }

    private static void register(ArtStylePreset preset) {
        PRESETS.put(preset.getKey(), preset);
    }

    /**
     * 根据 key 获取预设画风
     */
    public static ArtStylePreset getByKey(String key) {
        return PRESETS.get(key);
    }

    /**
     * 获取所有预设画风列表
     */
    public static List<ArtStylePreset> getAll() {
        return new ArrayList<>(PRESETS.values());
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ArtStylePreset {
        /** 画风唯一标识 */
        private String key;
        /** 画风显示名称 */
        private String name;
        /** 画风描述（中文，用于视频生成 prompt 前缀） */
        private String description;
        /** 画风英文提示词（用于图片生成） */
        private String imagePrompt;
        /** 参考图相对路径（如 /api/art-styles/cg.jpg） */
        private String referenceImagePath;
        /** 参考图公网 URL（上传到 OSS 后由系统配置填充） */
        private String referenceImagePublicUrl;
    }
}
