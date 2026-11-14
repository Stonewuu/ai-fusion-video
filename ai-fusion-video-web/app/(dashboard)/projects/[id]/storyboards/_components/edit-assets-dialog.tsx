"use client";

import { useState, useEffect, useMemo } from "react";
import { Check, Loader2, MapPin, Package, Users, ZoomIn, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import { toastApiError } from "@/lib/api/toast-api-error";
import type { AssetWithItems } from "@/lib/api/asset";
import type { StoryboardItem } from "@/lib/api/storyboard";
import { SafeImage } from "@/components/media/managed-image";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

export interface ShotAssets {
  characterIds: number[];
  sceneAssetItemId: number;
  propIds: number[];
}

type AssetDraft = Partial<Pick<StoryboardItem, "shotNumber" | "autoShotNumber" | "characterIds" | "sceneAssetItemId" | "propIds">>;
type Choice = NonNullable<AssetWithItems["items"]>[number] & { assetName: string };

function AssetChoices({ title, icon: Icon, items, selected, onSelect, onPreview, single = false }: {
  title: string;
  icon: LucideIcon;
  items: Choice[];
  selected: Set<number>;
  onSelect: (id: number) => void;
  onPreview: (item: Choice) => void;
  single?: boolean;
}) {
  return (
    <section className="flex flex-col gap-3" aria-label={title}>
      <h4 className="flex items-center gap-2 text-sm font-medium"><Icon className="size-4 text-muted-foreground" />{title}</h4>
      {items.length === 0 ? <p className="text-sm text-muted-foreground">暂无匹配资产</p> : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3" role={single ? "radiogroup" : "group"} aria-label={title}>
          {items.map((item) => (
            <div key={item.id} className="relative min-w-0">
              <button type="button" role={single ? "radio" : "checkbox"} aria-checked={selected.has(item.id)}
                onClick={() => onSelect(item.id)} title={`${item.assetName} · ${item.name}`}
                className={cn("flex w-full flex-col rounded-xl border bg-card text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                  selected.has(item.id) ? "border-primary bg-primary/5" : "border-border/30 hover:border-border/50")}>
                <div className="relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-t-xl bg-muted/30">
                  {item.imageUrl ? <SafeImage src={resolveMediaUrl(item.imageUrl)} alt={item.name || item.assetName} className="size-full object-cover" /> : <Icon className="size-6 text-muted-foreground" />}
                  {selected.has(item.id) && <span className="absolute right-2 top-2 rounded-full bg-primary p-1 text-primary-foreground"><Check className="size-3" /></span>}
                </div>
                <span className="flex min-w-0 w-full flex-col gap-1 p-3">
                  <span className="truncate text-sm font-medium">{item.assetName}</span>
                  <span className="truncate text-xs text-muted-foreground">{item.name === item.assetName ? "基础设定" : item.name}</span>
                </span>
              </button>
              {item.imageUrl && <div className="absolute left-2 top-2"><Button variant="secondary" size="icon-xs" aria-label={`预览${item.name || item.assetName}`} onClick={() => onPreview(item)}><ZoomIn /></Button></div>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export function EditItemAssetsDialog({ open, onClose, item, assetsList, onConfirm, creating = false }: {
  open: boolean;
  onClose: () => void;
  item: AssetDraft | null;
  assetsList: AssetWithItems[];
  onConfirm: (data: ShotAssets) => Promise<void> | void;
  creating?: boolean;
}) {
  const [characters, setCharacters] = useState<Set<number>>(new Set());
  const [sceneId, setSceneId] = useState<number | null>(null);
  const [props, setProps] = useState<Set<number>>(new Set());
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<Choice | null>(null);
  useEffect(() => {
    if (open && item) {
      setCharacters(new Set(item.characterIds ?? []));
      setSceneId(item.sceneAssetItemId ?? null);
      setProps(new Set(item.propIds ?? []));
      setSearch("");
    }
  }, [open, item]);
  const groups = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    const choices = (type: string) => assetsList.filter((asset) => asset.type === type).flatMap((asset) =>
      (asset.items ?? []).map((entry) => ({ ...entry, assetName: asset.name })))
      .filter((entry) => `${entry.assetName} ${entry.name}`.toLocaleLowerCase().includes(query));
    return { scenes: choices("scene"), characters: choices("character"), props: choices("prop") };
  }, [assetsList, search]);
  const hasSelectedScene = assetsList.some((asset) => asset.type === "scene" && asset.items?.some((entry) => entry.id === sceneId));
  const toggle = (values: Set<number>, id: number) => {
    const next = new Set(values);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    return next;
  };
  const save = async () => {
    if (sceneId === null || !hasSelectedScene) {
      return;
    }
    setLoading(true);
    try {
      await onConfirm({ characterIds: [...characters], sceneAssetItemId: sceneId, propIds: [...props] });
      onClose();
    } catch (error) {
      toastApiError(error, creating ? "添加镜头失败" : "保存关联资产失败");
    } finally {
      setLoading(false);
    }
  };
  return (
    <>
      <Dialog open={open && item !== null} onOpenChange={(value) => {
        if (!value && !loading) {
          onClose();
        }
      }}>
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
          <DialogHeader className="shrink-0">
            <DialogTitle>{creating ? "添加镜头" : "关联镜头资产"}</DialogTitle>
            <DialogDescription>{creating ? "选择场景后添加镜头。" : `镜头 ${item?.shotNumber || item?.autoShotNumber || "—"}`}</DialogDescription>
          </DialogHeader>
          <div className="shrink-0 p-1"><Input aria-label="搜索资产" placeholder="搜索资产" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
          <DialogScrollBody className="flex flex-col gap-6">
            <AssetChoices title="场景（必选）" icon={MapPin} items={groups.scenes} selected={new Set(sceneId === null ? [] : [sceneId])} onSelect={setSceneId} onPreview={setPreview} single />
            <AssetChoices title="人物" icon={Users} items={groups.characters} selected={characters} onSelect={(id) => setCharacters(toggle(characters, id))} onPreview={setPreview} />
            <AssetChoices title="道具" icon={Package} items={groups.props} selected={props} onSelect={(id) => setProps(toggle(props, id))} onPreview={setPreview} />
          </DialogScrollBody>
          <DialogFooter className="shrink-0 items-center gap-2">
            {!hasSelectedScene && <span className="text-sm text-muted-foreground sm:mr-auto">请选择一个场景</span>}
            <Button variant="outline" onClick={onClose} disabled={loading}>取消</Button>
            <Button onClick={save} disabled={loading || !hasSelectedScene}>{loading && <Loader2 className="animate-spin" />}{creating ? "添加" : "保存"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={preview !== null} onOpenChange={(value) => {
        if (!value) {
          setPreview(null);
        }
      }}>
        <DialogContent className="max-h-[calc(100vh-2rem)] sm:max-w-3xl">
          <DialogHeader><DialogTitle>{preview?.name || preview?.assetName}</DialogTitle></DialogHeader>
          {preview?.imageUrl && <SafeImage src={resolveMediaUrl(preview.imageUrl)} alt={preview.name || preview.assetName} className="max-h-[70vh] w-full rounded-lg object-contain" />}
        </DialogContent>
      </Dialog>
    </>
  );
}
