package com.stonewu.fusion.service.ai;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 模型预设服务
 * <p>
 * 从 resources/model-presets/{vendor}/{category}.json 加载已知模型的默认配置。
 * 目录结构示例：
 * <pre>
 * model-presets/
 *   volcengine/
 *     image.json      ← 火山引擎图片模型
 *     video.json      ← 火山引擎视频模型（未来）
 *   openai/
 *     image.json
 *   vertex_ai/
 *     image.json
 *   google_flow_reverse_api/
 *     image.json
 *     video.json
 * </pre>
 * <p>
 * 预设可作为显式的模型能力模板；选择后由 capabilityPresetCode 引用，模型 config 仅保存覆盖项。
 */
@Service
@Slf4j
public class ModelPresetService {

    /** code → 预设 JSON 对象 */
    private final Map<String, JSONObject> presets = new LinkedHashMap<>();

    /** 实际模型代码 + 类型 → 能力预设代码；用于同一上游代码跨图片/视频复用的场景。 */
    private final Map<String, String> presetCodesByModelKey = new LinkedHashMap<>();

    @PostConstruct
    public void init() {
        loadAllPresets();
        log.info("[ModelPreset] 已加载 {} 个模型预设: {}", presets.size(), presets.keySet());
    }

    /**
     * 扫描 model-presets 目录下所有子目录中的 JSON 文件
     */
    private void loadAllPresets() {
        try {
            PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
            Resource[] resources = resolver.getResources("classpath:model-presets/**/*.json");
            for (Resource resource : resources) {
                loadPresetResource(resource);
            }
        } catch (IOException e) {
            log.error("[ModelPreset] 扫描预设目录失败", e);
        }
    }

    private void loadPresetResource(Resource resource) {
        try {
            String json = resource.getContentAsString(StandardCharsets.UTF_8);
            JSONArray array = JSONUtil.parseArray(json);
            for (int i = 0; i < array.size(); i++) {
                JSONObject obj = array.getJSONObject(i);
                String code = obj.getStr("code");
                if (StrUtil.isNotBlank(code)) {
                    presets.put(code, obj);
                    String modelCode = StrUtil.blankToDefault(obj.getStr("modelCode"), code);
                    Integer modelType = obj.getInt("modelType");
                    if (modelType != null && StrUtil.isNotBlank(modelCode)) {
                        presetCodesByModelKey.put(modelKey(modelCode, modelType), code);
                    }
                }
            }
            log.debug("[ModelPreset] 加载预设文件: {}, 模型数: {}", resource.getFilename(), array.size());
        } catch (IOException e) {
            log.error("[ModelPreset] 加载预设文件失败: {}", resource.getDescription(), e);
        }
    }

    /**
     * 根据能力预设代码查找预设
     */
    public JSONObject getPreset(String code) {
        return presets.get(code);
    }

    /**
     * 根据能力预设代码获取预设的 config JSON 字符串
     */
    public String getPresetConfig(String code) {
        JSONObject preset = presets.get(code);
        if (preset == null) {
            return null;
        }
        JSONObject config = preset.getJSONObject("config");
        return config != null ? config.toString() : null;
    }

    /**
     * 判断代码是否精确对应内置能力预设。
     */
    public boolean hasPreset(String code) {
        return presets.containsKey(code);
    }

    /**
     * 按上游实际模型代码和模型类型查找能力预设代码。
     */
    public String findPresetCode(String modelCode, Integer modelType) {
        if (StrUtil.isBlank(modelCode) || modelType == null) {
            return null;
        }
        JSONObject directPreset = presets.get(modelCode);
        if (directPreset != null && modelType.equals(directPreset.getInt("modelType"))) {
            return modelCode;
        }
        return presetCodesByModelKey.get(modelKey(modelCode, modelType));
    }

    /**
     * 获取所有预设列表（供前端展示可选模型）
     */
    public List<JSONObject> getAllPresets() {
        return List.copyOf(presets.values());
    }

    /**
     * 按模型类型获取预设列表
     *
     * @param modelType 模型类型：2-图片生成 3-视频生成
     */
    public List<JSONObject> getPresetsByType(int modelType) {
        return presets.values().stream()
                .filter(p -> p.getInt("modelType", 0) == modelType)
                .toList();
    }

    private String modelKey(String modelCode, int modelType) {
        return modelCode.trim() + "\u0000" + modelType;
    }
}
