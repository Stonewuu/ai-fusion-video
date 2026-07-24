"use client";

import { useEffect, useState } from "react";
import { ImageIcon, Loader2, PackagePlus, Video } from "lucide-react";
import { toast } from "sonner";
import { assetApi, type AssetWithItems } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { projectApi, type Project } from "@/lib/api/project";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { SafeImage } from "@/components/ui/safe-image";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { GenerationAssetTarget } from "./generation-types";
import { getResultMediaUrl, getResultPreviewUrl } from "./generation-utils";

interface GenerationAssetDialogProps {
  target: GenerationAssetTarget;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAttached: () => void;
}

const ASSET_TYPES = [
  { value: "character", label: "角色" },
  { value: "scene", label: "场景" },
  { value: "prop", label: "道具" },
];

export function GenerationAssetDialog({
  target,
  open,
  onOpenChange,
  onAttached,
}: GenerationAssetDialogProps) {
  const [attachMode, setAttachMode] = useState<"existing" | "new">("existing");
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [assets, setAssets] = useState<AssetWithItems[]>([]);
  const [assetId, setAssetId] = useState<number | null>(null);
  const [assetType, setAssetType] = useState(
    target.mode === "video" ? "scene" : "character",
  );
  const [masterName, setMasterName] = useState("");
  const [setting, setSetting] = useState(target.prompt);
  const [itemName, setItemName] = useState(
    target.mode === "image" ? "生成图片" : "生成视频",
  );
  const [appearance, setAppearance] = useState(target.prompt);
  const [loadingProjects, setLoadingProjects] = useState(true);
  const [loadingAssets, setLoadingAssets] = useState(false);
  const [saving, setSaving] = useState(false);

  const sourceMediaUrl = getResultMediaUrl(target.item) || "";
  const previewUrl = getResultPreviewUrl(target.item) || sourceMediaUrl;

  const loadAssets = async (nextProjectId: number) => {
    setLoadingAssets(true);
    try {
      const list = await assetApi.listWithItems(nextProjectId);
      setAssets(list);
      setAssetId(list[0]?.id ?? null);
      if (list.length === 0) setAttachMode("new");
    } catch {
      setAssets([]);
      setAssetId(null);
    } finally {
      setLoadingAssets(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    projectApi.list()
      .then((list) => {
        if (cancelled) return;
        setProjects(list);
        const firstProjectId = list[0]?.id ?? null;
        setProjectId(firstProjectId);
        if (firstProjectId) void loadAssets(firstProjectId);
      })
      .catch(() => {
        if (!cancelled) {
          setProjects([]);
          setProjectId(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingProjects(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const handleProjectChange = (nextProjectId: number | null) => {
    setProjectId(nextProjectId);
    setAssetId(null);
    setAssets([]);
    if (nextProjectId !== null) void loadAssets(nextProjectId);
  };

  const save = async () => {
    if (!projectId || !itemName.trim()) return;
    if (attachMode === "existing" && !assetId) return;
    if (attachMode === "new" && !masterName.trim()) return;

    setSaving(true);
    try {
      let targetAssetId = assetId;
      let sortOrder = 0;

      if (attachMode === "new") {
        const created = await assetApi.create({
          projectId,
          type: assetType,
          name: masterName.trim(),
          description: setting.trim(),
          properties: JSON.stringify({ setting: setting.trim() }),
        });
        targetAssetId = created.id;
      } else {
        sortOrder = assets.find((asset) => asset.id === assetId)?.items.length || 0;
      }

      if (!targetAssetId) return;
      await assetApi.createItem({
        assetId: targetAssetId,
        itemType: target.mode === "image" ? "generated_image" : "generated_video",
        name: itemName.trim(),
        imageUrl:
          target.mode === "image"
            ? sourceMediaUrl || undefined
            : previewUrl || undefined,
        properties: JSON.stringify({
          appearanceDescription: appearance.trim(),
          sourceGenerationMode: target.mode,
          sourceMediaUrl,
          sourceGenerationItemId: target.item.id,
        }),
        sortOrder,
      });

      toast.success(attachMode === "new" ? "资产已创建" : "已添加到资产");
      onAttached();
      onOpenChange(false);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  const canSave =
    Boolean(projectId && itemName.trim()) &&
    (attachMode === "existing" ? Boolean(assetId) : Boolean(masterName.trim()));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden p-0 sm:max-w-4xl">
        <DialogHeader className="shrink-0 border-b border-border/40 px-6 py-5">
          <DialogTitle className="flex items-center gap-2 text-lg">
            <PackagePlus className="h-4 w-4 text-primary" />
            添加到资产
          </DialogTitle>
          <DialogDescription>保存为新资产，或添加到已有资产。</DialogDescription>
        </DialogHeader>

        <div className="grid min-h-0 flex-1 overflow-y-auto lg:grid-cols-[280px_minmax(0,1fr)]">
          <div className="border-b border-border/40 bg-muted/15 p-5 lg:border-b-0 lg:border-r">
            <div className="overflow-hidden rounded-2xl border border-border/45 bg-background">
              {target.mode === "image" ? (
                <SafeImage
                  src={resolveMediaUrl(previewUrl) || undefined}
                  alt="生成结果"
                  className="aspect-square w-full object-contain"
                  fallbackType="image"
                />
              ) : sourceMediaUrl ? (
                <video
                  controls
                  preload="metadata"
                  poster={resolveMediaUrl(previewUrl) || undefined}
                  src={resolveMediaUrl(sourceMediaUrl) || undefined}
                  className="aspect-video w-full bg-black object-contain"
                />
              ) : (
                <div className="grid aspect-video place-items-center text-muted-foreground">
                  <Video className="h-7 w-7" />
                </div>
              )}
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
              {target.mode === "image" ? (
                <ImageIcon className="h-3.5 w-3.5" />
              ) : (
                <Video className="h-3.5 w-3.5" />
              )}
              {target.mode === "image" ? "图片结果" : "视频结果"}
            </div>
          </div>

          <div className="min-w-0 p-5">
            <div className="grid grid-cols-2 rounded-xl bg-muted/55 p-1">
              <button
                type="button"
                onClick={() => setAttachMode("existing")}
                className={cn(
                  "h-8 rounded-lg text-xs font-medium transition-all",
                  attachMode === "existing"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                已有资产
              </button>
              <button
                type="button"
                onClick={() => setAttachMode("new")}
                className={cn(
                  "h-8 rounded-lg text-xs font-medium transition-all",
                  attachMode === "new"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                新建资产
              </button>
            </div>

            <div className="mt-4 space-y-3.5">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">项目</label>
                <Select
                  value={projectId}
                  onValueChange={(value) =>
                    handleProjectChange(value === null ? null : Number(value))
                  }
                  items={projects.map((project) => ({
                    value: project.id,
                    label: project.name,
                  }))}
                >
                  <SelectTrigger className="h-9 w-full rounded-xl">
                    <SelectValue
                      placeholder={loadingProjects ? "加载项目…" : "选择项目"}
                    />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {projects.map((project) => (
                        <SelectItem key={project.id} value={project.id}>
                          {project.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>

              {attachMode === "existing" ? (
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">
                    主资产
                  </label>
                  <Select
                    value={assetId}
                    onValueChange={(value) =>
                      setAssetId(value === null ? null : Number(value))
                    }
                    items={assets.map((asset) => ({
                      value: asset.id,
                      label: asset.name,
                    }))}
                  >
                    <SelectTrigger className="h-9 w-full rounded-xl">
                      <SelectValue
                        placeholder={loadingAssets ? "加载资产…" : "选择主资产"}
                      />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {assets.map((asset) => (
                          <SelectItem key={asset.id} value={asset.id}>
                            {asset.name}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_130px]">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground">
                      主资产名称
                    </label>
                    <Input
                      value={masterName}
                      onChange={(event) => setMasterName(event.target.value)}
                      placeholder="资产名称"
                      className="h-9 rounded-lg"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground">
                      类型
                    </label>
                    <Select
                      value={assetType}
                      onValueChange={(value) => setAssetType(String(value))}
                      items={ASSET_TYPES}
                    >
                      <SelectTrigger className="h-9 w-full rounded-xl">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {ASSET_TYPES.map((type) => (
                            <SelectItem key={type.value} value={type.value}>
                              {type.label}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {attachMode === "new" && (
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">
                    资产设定
                  </label>
                  <Textarea
                    value={setting}
                    onChange={(event) => setSetting(event.target.value)}
                    rows={3}
                    className="rounded-lg text-sm leading-5"
                  />
                </div>
              )}

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  子资产名称
                </label>
                <Input
                  value={itemName}
                  onChange={(event) => setItemName(event.target.value)}
                  placeholder="子资产名称"
                  className="h-9 rounded-lg"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  外观描写
                </label>
                <Textarea
                  value={appearance}
                  onChange={(event) => setAppearance(event.target.value)}
                  rows={4}
                  className="rounded-lg text-sm leading-5"
                />
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="shrink-0 border-t border-border/40 px-6 py-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button disabled={!canSave || saving} onClick={() => void save()}>
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <PackagePlus className="h-4 w-4" />
            )}
            {attachMode === "new" ? "创建并添加" : "添加"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
