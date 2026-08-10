"use client";

import { useState, useEffect, useCallback } from "react";
import {
  X,
  Check,
  AlertTriangle,
  Upload,
  Loader2,
  ImageIcon,
  Palette,
  Type,
  Monitor,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { getApiErrorMessage, readApiResponseError } from "@/lib/api/api-error";
import { projectApi, type ProjectCreateReq } from "@/lib/api/project";
import { artStyleApi } from "@/lib/api/art-style";
import type { ArtStylePreset } from "@/lib/api/art-style";
import { storageConfigApi, uploadFile } from "@/lib/api/storage";
import { resolveMediaUrl, http } from "@/lib/api/client";

// 项目类型选项
const projectTypes = ["漫剧", "短剧", "动画", "纪录片", "宣传片", "MV"];

// 画面比例选项
const aspectRatios = [
  { label: "16:9", desc: "横屏" },
  { label: "9:16", desc: "竖屏" },
  { label: "1:1", desc: "方形" },
  { label: "4:3", desc: "传统" },
];

interface CreateProjectDialogProps {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
}

export function CreateProjectDialog({
  open,
  onClose,
  onCreated,
}: CreateProjectDialogProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // 项目配置
  const [projectType, setProjectType] = useState(projectTypes[0]);
  const [aspectRatio, setAspectRatio] = useState(aspectRatios[0].label);

  // 画风
  const [presets, setPresets] = useState<ArtStylePreset[]>([]);
  const [artStyle, setArtStyle] = useState<string>("");
  const [presetsLoading, setPresetsLoading] = useState(false);
  const [uploadingPreset, setUploadingPreset] = useState<string | null>(null);

  // 存储 & 外网访问状态
  const [hasStorage, setHasStorage] = useState(false);
  const [hasExternalAccess, setHasExternalAccess] = useState(false);

  // 对话框打开时加载数据
  useEffect(() => {
    if (!open) return;

    // 加载画风预设，默认选第一个
    setPresetsLoading(true);
    artStyleApi
      .getPresets()
      .then((data) => {
        setPresets(data);
        if (data.length > 0) setArtStyle(data[0].key);
      })
      .catch((error) => {
        console.error("加载画风预设失败:", error);
        toast.error(getApiErrorMessage(error));
      })
      .finally(() => setPresetsLoading(false));

    // 加载存储配置 + 系统配置，判断上传 & 外网能力
    storageConfigApi
      .list()
      .then((configs) => {
        setHasStorage(configs.length > 0);
        const hasPublicStorage = configs.some((c) => c.type !== "local");
        http
          .get<never, { configKey: string; configValue: string }[]>(
            "/api/system/config"
          )
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
      })
      .catch((error) => {
        console.error("加载存储配置失败:", error);
        toast.error(getApiErrorMessage(error));
      });
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // 关闭时重置表单
  const handleClose = useCallback(() => {
    setName("");
    setDescription("");
    setError("");
    setProjectType(projectTypes[0]);
    setAspectRatio(aspectRatios[0].label);
    setArtStyle("");
    setPresets([]);
    onClose();
  }, [onClose]);

  const handleSubmit = async () => {
    if (!name.trim()) {
      setError("请输入项目名称");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data: ProjectCreateReq = {
        name: name.trim(),
        properties: JSON.stringify({ type: projectType, aspectRatio }),
      };
      if (description.trim()) data.description = description.trim();

      const created = await projectApi.create(data);

      // 若选择了画风则单独更新（创建接口不含画风字段）
      if (artStyle) {
        await projectApi.update({ id: created.id, artStyle });
      }

      onCreated();
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setLoading(false);
    }
  };

  // 上传预设参考图到存储（全局系统配置，与项目设置页行为一致）
  const handleUploadPresetImage = useCallback(
    async (preset: ArtStylePreset) => {
      if (!preset.referenceImagePath || !hasStorage) return;
      setUploadingPreset(preset.key);
      try {
        const imgUrl = resolveMediaUrl(preset.referenceImagePath);
        if (!imgUrl) return;
        const resp = await fetch(imgUrl);
        if (!resp.ok) throw new Error(await readApiResponseError(resp));
        const blob = await resp.blob();
        const file = new File([blob], `${preset.key}.png`, {
          type: blob.type || "image/png",
        });
        const url = await uploadFile(file, "art-styles");
        await http.put("/api/system/config", {
          [`art_preset_url:${preset.key}`]: url,
        });
        const updatedPresets = await artStyleApi.getPresets();
        setPresets(updatedPresets);
      } catch (err) {
        console.error("上传预设参考图失败:", err);
        toast.error(getApiErrorMessage(err));
      } finally {
        setUploadingPreset(null);
      }
    },
    [hasStorage]
  );

  const selectedPreset = presets.find((p) => p.key === artStyle);
  const isPresetRefAvailable =
    (selectedPreset?.referenceImagePublicUrl?.length ?? 0) > 0;

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* 遮罩 */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="modal-overlay fixed inset-0 z-50"
            onClick={handleClose}
          />
          {/* 弹窗 */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl"
          >
            <div
              className={cn(
                "modal-surface rounded-2xl border border-border/40",
                "flex flex-col overflow-hidden max-h-[90vh]"
              )}
            >
              {/* 标题栏 */}
              <div className="flex items-center justify-between px-6 pt-6 pb-4 shrink-0">
                <h2 className="text-lg font-semibold">新建项目</h2>
                <button
                  onClick={handleClose}
                  className="p-1.5 rounded-lg hover:bg-muted transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              </div>

              {/* 可滚动内容区 */}
              <div className="overflow-y-auto flex-1 px-6 space-y-5 pb-2">
                {/* 基本信息 */}
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-1.5">
                      项目名称 <span className="text-destructive">*</span>
                    </label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="例如：品牌宣传片"
                      className={cn(
                        "w-full px-3.5 py-2.5 rounded-xl text-sm",
                        "bg-muted/50 border border-border/40",
                        "focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50",
                        "placeholder:text-muted-foreground/50 transition-all"
                      )}
                      autoFocus
                      onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1.5">
                      项目描述
                    </label>
                    <textarea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="简要描述项目内容和目标..."
                      rows={2}
                      className={cn(
                        "w-full px-3.5 py-2.5 rounded-xl text-sm resize-none",
                        "bg-muted/50 border border-border/40",
                        "focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50",
                        "placeholder:text-muted-foreground/50 transition-all"
                      )}
                    />
                  </div>
                </div>

                {/* 分隔线 */}
                <div className="border-t border-border/20" />

                {/* 项目类型 */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Type className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">项目类型</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {projectTypes.map((t) => (
                      <button
                        key={t}
                        type="button"
                        onClick={() => setProjectType(t)}
                        className={cn(
                          "px-3 py-1.5 rounded-lg text-sm font-medium transition-all",
                          projectType === t
                            ? "bg-primary text-primary-foreground shadow-sm"
                            : "bg-muted/50 text-muted-foreground hover:bg-muted/80 border border-border/30 hover:border-primary/40"
                        )}
                      >
                        {projectType === t && (
                          <Check className="h-3 w-3 inline-block mr-1 -mt-0.5" />
                        )}
                        {t}
                      </button>
                    ))}
                  </div>
                </div>

                {/* 画面比例 */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Monitor className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">画面比例</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {aspectRatios.map((ar) => (
                      <button
                        key={ar.label}
                        type="button"
                        onClick={() => setAspectRatio(ar.label)}
                        className={cn(
                          "flex flex-col items-center gap-0.5 px-5 py-2 rounded-xl text-sm font-medium transition-all",
                          aspectRatio === ar.label
                            ? "bg-primary text-primary-foreground shadow-sm"
                            : "bg-muted/50 text-muted-foreground hover:bg-muted/80 border border-border/30 hover:border-primary/40"
                        )}
                      >
                        <span className="font-semibold text-xs">{ar.label}</span>
                        <span className="text-[10px] opacity-70">{ar.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* 画风 */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Palette className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">画风</span>
                  </div>

                  {presetsLoading ? (
                    <div className="flex items-center justify-center py-6">
                      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                    </div>
                  ) : presets.length === 0 ? (
                    <p className="text-xs text-muted-foreground py-3">
                      暂无可用预设画风
                    </p>
                  ) : (
                    <div className="space-y-3">
                      {/* 预设网格 */}
                      <div className="grid grid-cols-6 gap-2">
                        {presets.map((preset) => (
                          <button
                            key={preset.key}
                            type="button"
                            onClick={() =>
                              setArtStyle(
                                artStyle === preset.key ? "" : preset.key
                              )
                            }
                            className={cn(
                              "relative flex flex-col items-center gap-1.5 p-2 rounded-xl transition-all",
                              artStyle === preset.key
                                ? "bg-primary/10 border-2 border-primary shadow-sm"
                                : "bg-muted/30 border border-border/30 hover:border-primary/40 hover:bg-muted/50"
                            )}
                          >
                            <div className="w-full aspect-square rounded-lg overflow-hidden bg-muted/50 relative">
                              {preset.referenceImagePath ? (
                                <img
                                  src={
                                    resolveMediaUrl(preset.referenceImagePath) ||
                                    ""
                                  }
                                  alt={preset.name}
                                  className="w-full h-full object-cover"
                                />
                              ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                  <ImageIcon className="h-5 w-5 text-muted-foreground/40" />
                                </div>
                              )}
                              {artStyle === preset.key && (
                                <div className="absolute top-1 right-1 w-4 h-4 rounded-full bg-primary flex items-center justify-center">
                                  <Check className="h-2.5 w-2.5 text-primary-foreground" />
                                </div>
                              )}
                            </div>
                            <span
                              className={cn(
                                "text-[10px] font-medium",
                                artStyle === preset.key
                                  ? "text-primary"
                                  : "text-muted-foreground"
                              )}
                            >
                              {preset.name}
                            </span>
                          </button>
                        ))}
                      </div>

                      {/* 选中画风：描述 + 参考图状态 + 上传按钮 */}
                      {selectedPreset && (
                        <div className="p-3 rounded-lg border border-border/20 bg-muted/10 space-y-2.5">
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {selectedPreset.description}
                          </p>

                          <div className="flex items-center justify-between gap-3">
                            {isPresetRefAvailable ? (
                              <span className="flex items-center gap-1 text-[10px] text-emerald-500 font-medium">
                                <Check className="h-3 w-3" />
                                参考图已上传至存储
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
                            <button
                              type="button"
                              onClick={() =>
                                handleUploadPresetImage(selectedPreset)
                              }
                              disabled={
                                uploadingPreset === selectedPreset.key ||
                                !hasStorage
                              }
                              title={
                                !hasStorage
                                  ? "请先在系统设置中配置存储"
                                  : undefined
                              }
                              className={cn(
                                "flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium transition-all shrink-0",
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
                              {uploadingPreset === selectedPreset.key
                                ? "上传中…"
                                : isPresetRefAvailable
                                ? "重新上传"
                                : "上传到存储"}
                            </button>
                          </div>

                          {/* 警告：未配置存储 */}
                          {!hasStorage && (
                            <div className="flex items-start gap-1.5 p-2 rounded-lg bg-amber-500/5 border border-amber-500/20">
                              <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0 mt-0.5" />
                              <p className="text-[10px] text-muted-foreground leading-relaxed">
                                未配置存储，无法上传参考图。请在
                                <strong>系统设置 → 存储</strong>
                                中配置后再上传。
                              </p>
                            </div>
                          )}

                          {/* 警告：未上传到存储桶且无外网访问 */}
                          {!isPresetRefAvailable && !hasExternalAccess && (
                            <div className="flex items-start gap-1.5 p-2 rounded-lg bg-amber-500/5 border border-amber-500/20">
                              <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0 mt-0.5" />
                              <p className="text-[10px] text-muted-foreground leading-relaxed">
                                参考图未上传到存储桶且未配置外网访问地址，AI API 将无法访问参考图。
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {error && <p className="text-sm text-destructive">{error}</p>}
              </div>

              {/* 底部按钮 */}
              <div className="flex justify-end gap-3 px-6 py-4 shrink-0 border-t border-border/20 mt-2">
                <button
                  onClick={handleClose}
                  className={cn(
                    "px-4 py-2 rounded-xl text-sm font-medium",
                    "hover:bg-muted transition-colors"
                  )}
                >
                  取消
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className={cn(
                    "px-5 py-2 rounded-xl text-sm font-medium",
                    "bg-primary text-primary-foreground",
                    "hover:opacity-90 active:scale-[0.98] transition-all",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  {loading ? "创建中..." : "创建项目"}
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
