"use client";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { SafeImage } from "@/components/media/managed-image";

import { useCommandRequestKey } from "@/hooks/use-command-request-key";

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
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { projectApi, type Project, type ProjectCreateReq } from "@/lib/api/project";
import { artStyleApi } from "@/lib/api/art-style";
import type { ArtStylePreset } from "@/lib/api/art-style";
import { storageConfigApi } from "@/lib/api/storage";
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
  onCreated: (project: Project) => void;
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
  const presetRequests = useCommandRequestKey();

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
        setHasStorage(configs.filter((config) => config.isDefault && config.status === 1).length === 1);
        http
          .get<never, string | null>(
            "/api/system/config/resource_base_url"
          )
          .then((resourceBaseUrl) => {
            setHasExternalAccess(!!resourceBaseUrl);
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
  }, [open]);

  // 关闭时重置表单
  const createRequest = useCommandRequestKey();
  const handleClose = useCallback(() => {
    if (loading) return;
    createRequest.clear();
    setName("");
    setDescription("");
    setError("");
    setProjectType(projectTypes[0]);
    setAspectRatio(aspectRatios[0].label);
    setArtStyle("");
    setPresets([]);
    onClose();
  }, [onClose, createRequest, loading]);

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
        properties: { type: projectType, aspectRatio },
        artStyle: artStyle || undefined,
      };
      if (description.trim()) data.description = description.trim();

      const project = await projectApi.create(data, createRequest.forPayload(data));
      createRequest.clear();

      onCreated(project);
      setName("");
      setDescription("");
      setError("");
      onClose();
    } catch (err) {
      setError(getApiErrorMessage(err));
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
        await artStyleApi.publish(preset.key, presetRequests.forPayload({ preset: preset.key }));
        const updatedPresets = await artStyleApi.getPresets();
        setPresets(updatedPresets);
      } catch (err) {
        console.error("上传预设参考图失败:", err);
        toast.error(getApiErrorMessage(err));
      } finally {
        setUploadingPreset(null);
      }
    },
    [hasStorage, presetRequests]
  );

  const selectedPreset = presets.find((p) => p.key === artStyle);
  const isPresetRefAvailable =
    (selectedPreset?.referenceImagePublicUrl?.length ?? 0) > 0;

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => { if (!nextOpen) handleClose(); }}>
      <DialogContent showCloseButton={false} className="max-h-[calc(100vh-2rem)] sm:max-w-2xl flex flex-col overflow-hidden">
              {/* 标题栏 */}
              <DialogHeader className="shrink-0 pr-8">
                <DialogTitle>新建项目</DialogTitle>
                <DialogDescription>设置作品名称与画面风格，稍后也可以在项目设置中调整。</DialogDescription>
              </DialogHeader>
              <Button variant="ghost" size="icon-sm" className="absolute top-4 right-4" aria-label="关闭对话框" disabled={loading} onClick={handleClose}><X /></Button>

              {/* 可滚动内容区 */}
              <div className="min-h-0 overflow-y-auto space-y-6 p-1">
                {/* 基本信息 */}
                <div className="space-y-4">
                  <div className="flex flex-col gap-2">
                    <label htmlFor="new-project-name" className="text-sm font-medium">
                      项目名称 <span className="text-destructive">*</span>
                    </label>
                    <Input
                      id="new-project-name"
                      type="text"
                      maxLength={200}
                      disabled={loading}
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="例如：品牌宣传片"
                      autoFocus
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.nativeEvent.isComposing && !loading) void handleSubmit(); }}
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <label htmlFor="new-project-description" className="text-sm font-medium">
                      项目描述
                    </label>
                    <Textarea
                      id="new-project-description"
                      disabled={loading}
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="简要描述故事内容和创作目标"
                      rows={2}
                      className="resize-none"
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
                        aria-pressed={projectType === t}
                        disabled={loading}
                        onClick={() => setProjectType(t)}
                        className={cn(
                          "px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
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
                        aria-pressed={aspectRatio === ar.label}
                        disabled={loading}
                        onClick={() => setAspectRatio(ar.label)}
                        className={cn(
                          "flex flex-col items-center gap-0.5 px-5 py-2 rounded-xl text-sm font-medium transition-colors",
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
                            aria-pressed={artStyle === preset.key}
                            disabled={loading}
                            onClick={() =>
                              setArtStyle(
                                artStyle === preset.key ? "" : preset.key
                              )
                            }
                            className={cn(
                              "relative flex flex-col items-center gap-1.5 p-2 rounded-xl transition-colors",
                              artStyle === preset.key
                                ? "bg-primary/10 border-2 border-primary shadow-sm"
                                : "bg-muted/30 border border-border/30 hover:border-primary/40 hover:bg-muted/50"
                            )}
                          >
                            <div className="w-full aspect-square rounded-lg overflow-hidden bg-muted/50 relative">
                              {preset.referenceImagePath ? (
                                <SafeImage
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
                                部分模型暂时无法读取参考图
                              </span>
                            )}
                            <Button type="button" variant="outline" size="xs"
                              onClick={() => handleUploadPresetImage(selectedPreset)}
                              disabled={uploadingPreset === selectedPreset.key || !hasStorage}
                              title={!hasStorage ? "请先配置默认媒体存储" : "登记内置参考图，供所有已登录用户使用"}>
                              {uploadingPreset === selectedPreset.key ? <Loader2 className="animate-spin" /> : <Upload />}
                              {uploadingPreset === selectedPreset.key ? "上传中…" : isPresetRefAvailable ? "重新上传" : "上传到存储"}
                            </Button>
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
                                支持直接读取图片的模型仍可使用。其他模型需要先上传参考图，或在系统设置中配置公网访问地址。
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
              </div>

              {/* 底部按钮 */}
              <DialogFooter className="shrink-0">
                <Button variant="outline" disabled={loading} onClick={handleClose}>
                  取消
                </Button>
                <Button
                  onClick={handleSubmit}
                  disabled={loading || !name.trim()}
                >
                  {loading ? "正在创建…" : "创建项目"}
                </Button>
              </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
