"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Settings,
  Palette,
  Monitor,
  Type,
  Trash2,
  Loader2,
  Check,
  Save,
  Upload,
  ImageIcon,
  X,
  AlertTriangle,
} from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { useProject } from "../project-context";
import { projectApi } from "@/lib/api/project";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { artStyleApi } from "@/lib/api/art-style";
import type { ArtStylePreset } from "@/lib/api/art-style";
import { storageConfigApi, uploadFile } from "@/lib/api/storage";
import { getApiErrorMessage, readApiResponseError } from "@/lib/api/api-error";
import { resolveMediaUrl, http } from "@/lib/api/client";
import { SafeImage } from "@/components/ui/safe-image";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};
const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

// 项目类型选项
const projectTypes = ["漫剧", "短剧", "动画", "纪录片", "宣传片", "MV"];

// 画面比例选项
const aspectRatios = [
  { label: "16:9", desc: "横屏" },
  { label: "9:16", desc: "竖屏" },
  { label: "1:1", desc: "方形" },
  { label: "4:3", desc: "传统" },
];

type ArtStyleTab = "preset" | "custom";

export default function ProjectSettingsPage() {
  const { confirm } = useConfirm();
  const router = useRouter();
  const { project, refresh } = useProject();

  // properties 中的简单配置
  const savedProperties = (project?.properties as Record<string, string>) || {};

  // 本地暂存
  const [propsDraft, setPropsDraft] = useState<Record<string, string>>({});
  const [artStyle, setArtStyle] = useState<string>("");
  const [artStyleDescription, setArtStyleDescription] = useState<string>("");
  const [artStyleImagePrompt, setArtStyleImagePrompt] = useState<string>("");
  const [artStyleImageUrl, setArtStyleImageUrl] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // 画风相关状态
  const [presets, setPresets] = useState<ArtStylePreset[]>([]);
  const [artStyleTab, setArtStyleTab] = useState<ArtStyleTab>("preset");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 加载预设画风
  useEffect(() => {
    artStyleApi.getPresets().then(setPresets).catch((error) => {
      console.error("加载画风预设失败:", error);
      toast.error(getApiErrorMessage(error));
    });
  }, []);

  // 当后端数据变化时，同步到本地
  useEffect(() => {
    if (!project) return;
    setPropsDraft({ ...savedProperties });
    setArtStyle(project.artStyle || "");
    setArtStyleDescription(project.artStyleDescription || "");
    setArtStyleImagePrompt(project.artStyleImagePrompt || "");
    setArtStyleImageUrl(project.artStyleImageUrl || "");
    setArtStyleTab(project.artStyle === "custom" ? "custom" : "preset");
  }, [project]); // eslint-disable-line react-hooks/exhaustive-deps

  // 是否有未保存的修改
  const hasChanges = useMemo(() => {
    if (!project) return false;
    // 检查 properties
    const keys = new Set([...Object.keys(propsDraft), ...Object.keys(savedProperties)]);
    for (const k of keys) {
      if ((propsDraft[k] || "") !== (savedProperties[k] || "")) return true;
    }
    // 检查画风字段
    if ((artStyle || "") !== (project.artStyle || "")) return true;
    if ((artStyleDescription || "") !== (project.artStyleDescription || "")) return true;
    if ((artStyleImagePrompt || "") !== (project.artStyleImagePrompt || "")) return true;
    if ((artStyleImageUrl || "") !== (project.artStyleImageUrl || "")) return true;
    return false;
  }, [propsDraft, savedProperties, artStyle, artStyleDescription, artStyleImagePrompt, artStyleImageUrl, project]);

  const handleToggleProp = (key: string, value: string) => {
    setPropsDraft((prev) => {
      const next = { ...prev };
      if (next[key] === value) {
        delete next[key];
      } else {
        next[key] = value;
      }
      return next;
    });
  };

  // 选择预设画风
  const handleSelectPreset = useCallback((preset: ArtStylePreset) => {
    if (artStyle === preset.key) {
      // 取消选中
      setArtStyle("");
    } else {
      // 切换到另一个预设，清空旧的上传地址（因为参考图不同）
      setArtStyle(preset.key);
    }
    setArtStyleImageUrl("");
    setArtStyleDescription("");
    setArtStyleImagePrompt("");
  }, [artStyle]);

  // 切换到自定义模式
  const handleSwitchToCustom = useCallback(() => {
    setArtStyleTab("custom");
    setArtStyle("custom");
  }, []);

  // 切换到预设模式
  const handleSwitchToPreset = useCallback(() => {
    setArtStyleTab("preset");
    setArtStyle("");
    setArtStyleDescription("");
    setArtStyleImagePrompt("");
    setArtStyleImageUrl("");
  }, []);

  // 上传自定义画风图片
  const handleUploadImage = useCallback(async (file: File) => {
    setUploading(true);
    try {
      const url = await uploadFile(file, "art-styles");
      setArtStyleImageUrl(url);
    } catch (err) {
      console.error("上传画风图片失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setUploading(false);
    }
  }, []);

  // 外网访问能力（后端资源公网地址或公有云存储）
  const [hasExternalAccess, setHasExternalAccess] = useState(false);
  const [hasStorage, setHasStorage] = useState(false);

  // 加载存储 & 系统配置，判断外网访问能力
  useEffect(() => {
    storageConfigApi.list().then((configs) => {
      setHasStorage(configs.length > 0);
      const hasPublicStorage = configs.some((c) => c.type !== "local");
      http.get<never, { configKey: string; configValue: string }[]>("/api/system/config")
        .then((list) => {
          const map: Record<string, string> = {};
          list.forEach((c) => { map[c.configKey] = c.configValue || ""; });
          setHasExternalAccess(
            hasPublicStorage || !!map.resource_base_url,
          );
        })
        .catch((error) => {
          console.error("加载系统配置失败:", error);
          toast.error(getApiErrorMessage(error));
        });
    }).catch((error) => {
      console.error("加载存储配置失败:", error);
      toast.error(getApiErrorMessage(error));
    });
  }, []);

  // 上传预设参考图到存储（保存到全局系统配置）
  const [uploadingPreset, setUploadingPreset] = useState<string | null>(null);

  const handleUploadPresetImage = useCallback(async (preset: ArtStylePreset) => {
    if (!preset.referenceImagePath) return;
    setUploadingPreset(preset.key);
    try {
      // 从后端静态资源获取图片 blob
      const imgUrl = resolveMediaUrl(preset.referenceImagePath);
      if (!imgUrl) return;
      const resp = await fetch(imgUrl);
      if (!resp.ok) throw new Error(await readApiResponseError(resp));
      const blob = await resp.blob();
      const file = new File([blob], `${preset.key}.png`, { type: blob.type || "image/png" });
      const url = await uploadFile(file, "art-styles");
      // 保存到系统配置（全局生效）
      await http.put("/api/system/config", { [`art_preset_url:${preset.key}`]: url });
      // 刷新预设列表以更新状态
      const updatedPresets = await artStyleApi.getPresets();
      setPresets(updatedPresets);
    } catch (err) {
      console.error("上传预设参考图失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setUploadingPreset(null);
    }
  }, []);

  // 自定义画风的参考图是否可用
  const isCustomRefAvailable = artStyleImageUrl?.startsWith("http://") || artStyleImageUrl?.startsWith("https://");


  const handleDeleteProject = async () => {
    if (!project) return;
    const ok = await confirm({ title: "彻底删除项目", description: "确定要删除该项目吗？此操作将永久移除所有剧本、分镜、资产等数据，不可恢复。", variant: "destructive", confirmText: "彻底删除项目" }); if (!ok) return;
    setDeleting(true);
    try {
      await projectApi.delete(project.id);
      router.push("/projects");
    } catch (err) {
      console.error("删除项目失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setDeleting(false);
    }
  };

  const handleSave = async () => {
    if (!project || !hasChanges) return;
    setSaving(true);
    try {
      // 更新 properties（JSON 字段）
      await projectApi.updateProperties(project.id, propsDraft);
      // 更新画风独立字段
      await projectApi.update({
        id: project.id,
        artStyle: artStyle || null,
        artStyleDescription: artStyleDescription || null,
        artStyleImagePrompt: artStyleImagePrompt || null,
        artStyleImageUrl: artStyleImageUrl || null,
      });
      await refresh();
    } catch (err) {
      console.error("保存设置失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  // 当前选中的预设
  const selectedPreset = presets.find((p) => p.key === artStyle);
  // 当前选中预设的参考图是否可用（基于预设全局 URL）
  const isPresetRefAvailable = selectedPreset?.referenceImagePublicUrl != null && selectedPreset.referenceImagePublicUrl.length > 0;

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible">
      {/* 标题 + 保存按钮 */}
      <motion.div variants={itemVariants} className="mb-6 flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Settings className="h-5 w-5 text-primary" />
          项目设置
        </h2>
        <button
          onClick={handleSave}
          disabled={!hasChanges || saving}
          className={cn(
            "flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium transition-all duration-200",
            hasChanges
              ? "bg-primary text-primary-foreground shadow-sm hover:opacity-90"
              : "bg-muted/50 text-muted-foreground cursor-not-allowed border border-border/30"
          )}
        >
          {saving ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Save className="h-4 w-4" />
          )}
          {saving ? "保存中…" : "保存设置"}
        </button>
      </motion.div>

      {/* 项目类型 */}
      <motion.div variants={itemVariants} className={cn(
        "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-5 mb-4"
      )}>
        <div className="flex items-center gap-2 mb-3">
          <Type className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">项目类型</h3>
        </div>
        <div className="flex flex-wrap gap-2">
          {projectTypes.map((t) => (
            <button
              key={t}
              onClick={() => handleToggleProp("type", t)}
              className={cn(
                "px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all",
                propsDraft.type === t
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted/50 text-muted-foreground hover:bg-muted/80 border border-border/30 hover:border-primary/40"
              )}
            >
              {propsDraft.type === t && (
                <Check className="h-3 w-3 inline-block mr-1 -mt-0.5" />
              )}
              {t}
            </button>
          ))}
        </div>
      </motion.div>

      {/* 画风 - 双模式 */}
      <motion.div variants={itemVariants} className={cn(
        "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-5 mb-4"
      )}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Palette className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">画风</h3>
          </div>
          {/* Tab 切换 */}
          <div className="flex gap-1 bg-muted/30 rounded-lg p-0.5">
            <button
              onClick={handleSwitchToPreset}
              className={cn(
                "px-3 py-1 rounded-md text-xs font-medium transition-all",
                artStyleTab === "preset"
                  ? "bg-background shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              预设
            </button>
            <button
              onClick={handleSwitchToCustom}
              className={cn(
                "px-3 py-1 rounded-md text-xs font-medium transition-all",
                artStyleTab === "custom"
                  ? "bg-background shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              自定义
            </button>
          </div>
        </div>

        {artStyleTab === "preset" ? (
          /* ===== 预设画风 ===== */
          <div>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-4">
              {presets.map((preset) => (
                <button
                  key={preset.key}
                  onClick={() => handleSelectPreset(preset)}
                  className={cn(
                    "group relative flex flex-col items-center gap-2 p-3 rounded-xl transition-all",
                    artStyle === preset.key
                      ? "bg-primary/10 border-2 border-primary shadow-sm"
                      : "bg-muted/30 border border-border/30 hover:border-primary/40 hover:bg-muted/50"
                  )}
                >
                  {/* 参考图缩略图 */}
                  <div className="w-full aspect-square rounded-lg overflow-hidden bg-muted/50 relative">
                    <SafeImage
                      src={resolveMediaUrl(preset.referenceImagePath)}
                      fallbackType="image"
                      alt={preset.name}
                      className="w-full h-full object-cover"
                    />
                    {artStyle === preset.key && (
                      <div className="absolute top-1 right-1 w-5 h-5 rounded-full bg-primary flex items-center justify-center z-10">
                        <Check className="h-3 w-3 text-primary-foreground" />
                      </div>
                    )}
                  </div>
                  <span className={cn(
                    "text-xs font-medium",
                    artStyle === preset.key ? "text-primary" : "text-muted-foreground"
                  )}>
                    {preset.name}
                  </span>
                </button>
              ))}
            </div>

            {/* 选中预设时显示描述 + 参考图状态 */}
            {selectedPreset && (
              <div className="space-y-3">
                <div className="p-3 rounded-lg bg-muted/20 border border-border/20">
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {selectedPreset.description}
                  </p>
                </div>

                {/* 参考图状态 */}
                <div className="p-3 rounded-lg border border-border/20 bg-muted/10">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ImageIcon className="h-4 w-4 text-muted-foreground" />
                      <span className="text-xs font-medium">参考图状态</span>
                    </div>
                    {isPresetRefAvailable ? (
                      <span className="flex items-center gap-1 text-[10px] text-emerald-500 font-medium">
                        <Check className="h-3 w-3" />
                        已上传至存储
                      </span>
                    ) : hasExternalAccess ? (
                      <span className="flex items-center gap-1 text-[10px] text-sky-500 font-medium">
                        <Check className="h-3 w-3" />
                        将通过外网地址提供给 AI
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-[10px] text-amber-500 font-medium">
                        <AlertTriangle className="h-3 w-3" />
                        未上传且未配置外网访问
                      </span>
                    )}
                  </div>

                  <div className="mt-2 space-y-2">
                    {!isPresetRefAvailable && !hasExternalAccess && (
                      <p className="text-[10px] text-muted-foreground leading-relaxed">
                        参考图未上传到存储桶且未配置外网访问，AI API 将无法访问参考图。
                        建议上传到存储，或在<strong>系统设置 → 通用</strong>中配置外网访问地址。
                      </p>
                    )}
                    {!isPresetRefAvailable && hasExternalAccess && (
                      <p className="text-[10px] text-muted-foreground leading-relaxed">
                        已配置外网访问地址，AI 将通过该地址访问本地参考图。
                        上传到存储可获得更稳定的访问效果。
                      </p>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleUploadPresetImage(selectedPreset); }}
                      disabled={uploadingPreset === selectedPreset.key || !hasStorage}
                      title={!hasStorage ? "请先在系统设置中配置存储" : undefined}
                      className={cn(
                        "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                        !hasStorage
                          ? "opacity-50 cursor-not-allowed bg-muted/30 text-muted-foreground border border-border/30"
                          : isPresetRefAvailable
                          ? "bg-muted/30 text-muted-foreground hover:bg-muted/50 border border-border/30 hover:text-foreground"
                          : "bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20"
                      )}
                    >
                      {uploadingPreset === selectedPreset.key ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Upload className="h-3 w-3" />
                      )}
                      {uploadingPreset === selectedPreset.key ? "上传中…" : isPresetRefAvailable ? "重新上传" : "上传到存储"}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* ===== 自定义画风 ===== */
          <div className="space-y-4">
            {/* 上传参考图 */}
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-2 block">
                风格参考图
              </label>
              <div className="flex items-start gap-4">
                {artStyleImageUrl ? (
                  <div className="relative w-32 h-32 rounded-xl overflow-hidden border border-border/30 group">
                    <SafeImage
                      src={resolveMediaUrl(artStyleImageUrl)}
                      fallbackType="image"
                      alt="风格参考"
                      className="w-full h-full object-cover"
                    />
                    <button
                      onClick={() => setArtStyleImageUrl("")}
                      className="absolute top-1 right-1 w-6 h-6 rounded-full bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-10"
                    >
                      <X className="h-3 w-3 text-white" />
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                    className={cn(
                      "w-32 h-32 rounded-xl border-2 border-dashed border-border/40 flex flex-col items-center justify-center gap-2",
                      "hover:border-primary/40 hover:bg-muted/30 transition-all",
                      "text-muted-foreground"
                    )}
                  >
                    {uploading ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Upload className="h-5 w-5" />
                    )}
                    <span className="text-[10px]">
                      {uploading ? "上传中…" : "上传图片"}
                    </span>
                  </button>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleUploadImage(file);
                    e.target.value = "";
                  }}
                />
              </div>
            </div>

            {/* 画风描述 */}
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-2 block">
                画风描述（中文，用于视频生成提示词前缀）
              </label>
              <textarea
                value={artStyleDescription}
                onChange={(e) => setArtStyleDescription(e.target.value)}
                placeholder="例如：水彩手绘风格画面，柔和的色彩过渡，纸张纹理质感，温暖的暖色调光影…"
                rows={3}
                className={cn(
                  "w-full px-3 py-2 rounded-xl text-sm",
                  "bg-muted/30 border border-border/30",
                  "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                  "placeholder:text-muted-foreground/40 resize-none"
                )}
              />
            </div>

            {/* 英文提示词 */}
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-2 block">
                画风英文提示词（用于图片生成，可选）
              </label>
              <textarea
                value={artStyleImagePrompt}
                onChange={(e) => setArtStyleImagePrompt(e.target.value)}
                placeholder="例如：Watercolor hand-painted style, soft color transitions, paper texture, warm lighting…"
                rows={3}
                className={cn(
                  "w-full px-3 py-2 rounded-xl text-sm",
                  "bg-muted/30 border border-border/30",
                  "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
                  "placeholder:text-muted-foreground/40 resize-none"
                )}
              />
            </div>

            {/* 参考图网络访问警告 */}
            <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
              <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
              <p className="text-xs text-muted-foreground leading-relaxed">
                参考图需要通过网络地址传递给 AI API 生成图片。
                如果您使用本地存储，请在<strong>系统设置 → 通用</strong>中配置后端资源公网地址，
                或使用对象存储（OSS）自动获得公网 URL。
              </p>
            </div>
          </div>
        )}
      </motion.div>

      {/* 画面比例 */}
      <motion.div variants={itemVariants} className={cn(
        "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-5 mb-4"
      )}>
        <div className="flex items-center gap-2 mb-3">
          <Monitor className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">画面比例</h3>
        </div>
        <div className="flex flex-wrap gap-3">
          {aspectRatios.map((ar) => (
            <button
              key={ar.label}
              onClick={() => handleToggleProp("aspectRatio", ar.label)}
              className={cn(
                "flex flex-col items-center gap-1 px-5 py-3 rounded-xl text-sm font-medium transition-all",
                propsDraft.aspectRatio === ar.label
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted/50 text-muted-foreground hover:bg-muted/80 border border-border/30 hover:border-primary/40"
              )}
            >
              <span className="font-semibold">{ar.label}</span>
              <span className="text-[10px] opacity-70">{ar.desc}</span>
            </button>
          ))}
        </div>
      </motion.div>

      {/* 危险操作 */}
      <motion.div variants={itemVariants} className={cn(
        "rounded-xl border border-destructive/20 bg-destructive/5 p-5 mt-8"
      )}>
        <div className="flex items-center gap-2 mb-3">
          <Trash2 className="h-4 w-4 text-destructive" />
          <h3 className="text-sm font-semibold text-destructive">危险操作</h3>
        </div>
        <p className="text-xs text-muted-foreground mb-3">
          删除项目将永久移除所有剧本、分镜、资产等数据，此操作不可恢复。
        </p>
        <button
          onClick={handleDeleteProject}
          disabled={deleting}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
            "border border-destructive/30 text-destructive",
            "hover:bg-destructive/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {deleting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Trash2 className="h-4 w-4" />
          )}
          {deleting ? "删除中…" : "删除项目"}
        </button>
      </motion.div>
    </motion.div>
  );
}
