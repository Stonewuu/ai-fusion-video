package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.config.ArtStylePresets;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.system.SystemConfigService;
import cn.hutool.core.util.StrUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 查询项目详情工具（get_project）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ProjectQueryToolExecutor implements ToolExecutor {

    private final ProjectService projectService;
    private final SystemConfigService systemConfigService;

    @Override
    public String getToolName() {
        return "get_project";
    }

    @Override
    public String getDisplayName() {
        return "查询项目详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询指定项目的详细信息，包括项目名称、描述、封面、创建时间，
                以及项目画风配置（artStyleInfo：画风描述、英文提示词、参考图URL）。
                仅能查询当前用户有权限访问的项目。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "projectId": {
                            "type": "number",
                            "description": "项目ID（必填）"
                        }
                    },
                    "required": ["projectId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long projectId = params.getLong("projectId");
            if (projectId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 projectId").toString();
            }

            Long userId = context.getUserId();

            Project project = projectService.getById(projectId);
            if (!projectService.canAccessProject(project, userId)) {
                return JSONUtil.createObj().set("status", "error")
                        .set("message", "无权访问该项目").toString();
            }

            JSONObject result = JSONUtil.createObj()
                    .set("projectId", project.getId())
                    .set("name", project.getName())
                    .set("description", project.getDescription())
                    .set("coverUrl", project.getCoverUrl())
                    .set("properties", project.getProperties())
                    .set("ownerType", project.getOwnerType())
                    .set("ownerId", project.getOwnerId())
                    .set("createTime", project.getCreateTime())
                    .set("updateTime", project.getUpdateTime());

            // 附加画风信息
            result.set("artStyleInfo", buildArtStyleInfo(project));

            return result.toString();
        } catch (Exception e) {
            log.error("查询项目失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }

    /**
     * 构建画风信息：从数据库字段读取，预设则补充静态信息，并通过 resolvePublicUrl 生成完整 URL
     */
    private JSONObject buildArtStyleInfo(Project project) {
        String artStyleKey = project.getArtStyle();
        JSONObject info = JSONUtil.createObj();

        if (StrUtil.isBlank(artStyleKey)) {
            info.set("hasArtStyle", false);
            info.set("description", "高质量精细画面");
            info.set("imagePrompt", "High quality, detailed illustration, professional artwork");
            return info;
        }

        info.set("hasArtStyle", true);
        info.set("styleKey", artStyleKey);

        String refImagePath;

        if (!"custom".equals(artStyleKey)) {
            // 预设画风
            ArtStylePresets.ArtStylePreset preset = ArtStylePresets.getByKey(artStyleKey);
            if (preset != null) {
                info.set("name", preset.getName());
                info.set("description", preset.getDescription());
                info.set("imagePrompt", preset.getImagePrompt());
                info.set("isPreset", true);
                // 优先从系统配置获取预设的公网 URL，否则回退到本地静态路径
                String presetPublicUrl = systemConfigService.getValue("art_preset_url:" + artStyleKey);
                refImagePath = StrUtil.isNotBlank(presetPublicUrl)
                        ? presetPublicUrl
                        : preset.getReferenceImagePath();
            } else {
                info.set("name", artStyleKey);
                info.set("description", "高质量精细画面");
                info.set("imagePrompt", "高质量精细画面，专业级插画，细节丰富");
                info.set("isPreset", false);
                refImagePath = null;
            }
        } else {
            // 自定义画风
            info.set("name", "自定义画风");
            info.set("description", StrUtil.blankToDefault(project.getArtStyleDescription(), "高质量精细画面"));
            info.set("imagePrompt", StrUtil.blankToDefault(project.getArtStyleImagePrompt(), "High quality, detailed illustration, professional artwork"));
            info.set("isPreset", false);
            refImagePath = project.getArtStyleImageUrl();
        }

        // 解析参考图为完整公网 URL
        String publicUrl = systemConfigService.resolvePublicUrl(refImagePath);
        info.set("referenceImageUrl", publicUrl);
        info.set("referenceImageAvailable", publicUrl != null);

        if (publicUrl == null && StrUtil.isNotBlank(refImagePath)) {
            info.set("referenceImageWarning", "参考图无法通过网络访问，请在系统设置中配置后端资源公网地址或启用对象存储");
        }

        return info;
    }
}
