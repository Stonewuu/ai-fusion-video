"use client";

import { useRef, useState } from "react";
import { Check, ChevronDown, Globe, Loader2, RefreshCw, Search, SlidersHorizontal } from "lucide-react";
import { PLATFORM_LABELS, type ModelPreset } from "@/lib/api/ai-model";
import { modelsDevApi, type ModelsDevEntry, type ModelsDevSearchResult } from "@/lib/api/models-dev";
import { findOfficialGooglePreset } from "@/lib/settings/models-dev-match";
import { cn } from "@/lib/utils";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ModelPresetDetails } from "./model-preset-details";

type Selection = { source: "builtin"; preset: ModelPreset } | { source: "online"; entry: ModelsDevEntry };
interface Props {
  selectedCode?: string;
  presets: ModelPreset[];
  allPresets: ModelPreset[];
  hasCustomConfig: boolean;
  modelCode: string;
  modelType: number;
  apiConfigId?: number;
  platform?: string | null;
  matchedName?: string;
  onSelectCustom: () => void;
  onApplyPreset: (preset: ModelPreset) => void;
  onApplyOnline: (entry: ModelsDevEntry, fetchedAt: string) => void;
}

export function CapabilityPresetSelect(props: Props) {
  const [open, setOpen] = useState(false);
  const [source, setSource] = useState<"builtin" | "online">("builtin");
  const [query, setQuery] = useState("");
  const [vendor, setVendor] = useState("all");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [result, setResult] = useState<ModelsDevSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [confirmReplace, setConfirmReplace] = useState(false);
  const requestId = useRef(0);
  const current = props.allPresets.find(preset => preset.code === props.selectedCode);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const localModels = props.presets.filter(preset => (preset.name + " " + (preset.modelCode || preset.code)).toLocaleLowerCase().includes(normalizedQuery));
  const vendorEntries = source === "builtin"
    ? localModels.map(preset => [preset.platform, PLATFORM_LABELS[preset.platform] || preset.platform] as const)
    : (result?.providers ?? []).map(provider => [provider.id, provider.name] as const);
  const vendorOptions = [{ value: "all", label: "全部厂商" }, ...Array.from(new Map(vendorEntries), ([value, label]) => ({ value, label }))];
  const candidates: Selection[] = source === "builtin"
    ? localModels.filter(preset => vendor === "all" || preset.platform === vendor).map(preset => ({ source: "builtin", preset }))
    : (result?.models ?? []).map(entry => ({ source: "online", entry }));
  const selectedPreset = selection?.source === "builtin" ? selection.preset
    : selection ? findOfficialGooglePreset(selection.entry, props.allPresets, props.platform) : undefined;

  const search = async (provider = vendor) => {
    const id = ++requestId.current;
    setLoading(true);
    setError("");
    setSelection(null);
    setConfirmReplace(false);
    try {
      const data = await modelsDevApi.search(query.trim(), props.modelType, props.apiConfigId, provider === "all" ? undefined : provider);
      if (id === requestId.current) {
        setResult(data);
      }
    } catch {
      if (id === requestId.current) {
        setError("暂时无法获取模型，请重试");
      }
    } finally {
      if (id === requestId.current) {
        setLoading(false);
      }
    }
  };
  const changeOpen = (value: boolean) => {
    requestId.current++;
    setOpen(value);
    if (value) {
      setSource("builtin");
      setQuery("");
      setVendor("all");
      setResult(null);
      setLoading(false);
      setError("");
      setConfirmReplace(false);
      setSelection(current ? { source: "builtin", preset: current } : null);
    }
  };
  const apply = () => {
    if (!selection) {
      return;
    }
    if (selection.source === "builtin" && props.hasCustomConfig && !confirmReplace) {
      setConfirmReplace(true);
      return;
    }
    try {
      if (selection.source === "builtin") {
        props.onApplyPreset(selection.preset);
      } else if (result) {
        props.onApplyOnline(selection.entry, result.fetchedAt);
      }
      changeOpen(false);
    } catch {
      setError("请先检查高级配置中的 JSON 格式");
    }
  };

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <Label className="text-xs text-muted-foreground">模型预设</Label>
      <Button variant="outline" className="w-full justify-between" onClick={() => changeOpen(true)}>
        <span className="flex min-w-0 items-center gap-2">
          {current ? <ModelVendorIcon source={current} /> : props.matchedName ? <Globe /> : <SlidersHorizontal />}
          <span className="truncate">{current?.name || props.matchedName || "选择预设"}</span>
        </span>
        <ChevronDown />
      </Button>
      <Dialog open={open} onOpenChange={changeOpen}>
        <DialogContent className="sm:max-w-3xl max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
          <DialogHeader className="shrink-0">
            <DialogTitle>选择模型预设</DialogTitle>
            <DialogDescription>搜索模型，预览参数后应用。</DialogDescription>
          </DialogHeader>
          <div className="flex shrink-0 items-center gap-2" role="group" aria-label="预设来源">
            {(["builtin", "online"] as const).map(value => <Button key={value} variant={source === value ? "secondary" : "ghost"}
              aria-pressed={source === value} onClick={() => {
                requestId.current++;
                setSource(value);
                if (value === "online" && !query.trim()) {
                  setQuery(props.modelCode);
                }
                setVendor("all");
                setSelection(null);
                setLoading(false);
                setError("");
                setConfirmReplace(false);
              }}>{value === "builtin" ? "内置预设" : "联网搜索"}</Button>)}
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2 p-1">
            <Input aria-label="搜索预设" placeholder="搜索模型名称或 ID" value={query} maxLength={120}
              className="min-w-0 flex-1 basis-full sm:basis-0" onChange={event => {
                setQuery(event.target.value);
                setVendor("all");
                setSelection(null);
                setConfirmReplace(false);
                if (source === "online") {
                  requestId.current++;
                  setResult(null);
                  setLoading(false);
                }
              }} onKeyDown={event => {
                if (event.key === "Enter" && source === "online" && !loading) {
                  event.preventDefault();
                  void search();
                }
              }} />
            {vendorOptions.length > 2 && <Select items={vendorOptions} value={vendor} onValueChange={value => {
              setVendor(String(value));
              if (source === "online") {
                void search(String(value));
              }
            }}>
              <SelectTrigger aria-label="筛选厂商" className="w-36 text-sm"><SelectValue placeholder="全部厂商" /></SelectTrigger>
              <SelectContent className="max-h-60 text-sm"><SelectGroup>{vendorOptions.map(option => <SelectItem key={option.value} value={option.value} className="text-sm">{option.label}</SelectItem>)}</SelectGroup></SelectContent>
            </Select>}
            {source === "online" && <Button variant="outline" disabled={loading} onClick={() => void search()}><Search />搜索</Button>}
          </div>
          <div className="grid min-h-0 gap-4 overflow-y-auto p-1 md:grid-cols-2">
            <div className="min-w-0">
              {loading ? <p className="flex items-center justify-center gap-2 py-12 text-muted-foreground"><Loader2 className="size-4 animate-spin" />正在搜索</p>
                : error ? <div className="flex flex-col items-center gap-3 py-8" role="alert"><p className="text-sm text-destructive">{error}</p><Button variant="outline" onClick={() => void search()}><RefreshCw />重试</Button></div>
                : source === "online" && !result ? <p className="py-12 text-center text-sm text-muted-foreground">输入关键词，搜索模型</p>
                : candidates.length === 0 ? <p className="py-12 text-center text-sm text-muted-foreground">没有找到预设，换个关键词试试</p>
                : <div className="flex max-h-80 flex-col gap-1 overflow-y-auto p-1">
                  {candidates.map(candidate => {
                    const id = candidate.source === "builtin" ? candidate.preset.code : candidate.entry.providerId + "/" + candidate.entry.modelId;
                    const name = candidate.source === "builtin" ? candidate.preset.name : candidate.entry.name;
                    const label = candidate.source === "builtin" ? candidate.preset.modelCode || candidate.preset.code : candidate.entry.providerName;
                    const selectedId = selection?.source === "builtin" ? selection.preset.code : selection ? selection.entry.providerId + "/" + selection.entry.modelId : null;
                    const active = selectedId === id;
                    return <button type="button" key={id} aria-pressed={active} onClick={() => {
                      setSelection(candidate);
                      setConfirmReplace(false);
                    }} className={cn("flex w-full items-center gap-3 rounded-xl border border-transparent p-3 text-left outline-none transition-colors motion-reduce:transition-none hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring",
                      active && "border-primary/30 bg-primary/5")}>
                      <span className="min-w-0 flex-1"><span className="block truncate text-sm font-medium" title={name}>{name}</span><span className="block truncate text-xs text-muted-foreground" title={label}>{label}</span></span>
                      {active && <Check className="size-4 shrink-0 text-primary" />}
                    </button>;
                  })}
                </div>}
            </div>
            <div className="min-w-0 border-t border-border/20 p-1 pt-4 md:border-t-0 md:border-l md:pt-1 md:pl-4">
              {selection ? <ModelPresetDetails preset={selectedPreset} online={selection.source === "online" ? selection.entry : undefined} />
                : <p className="py-8 text-center text-sm text-muted-foreground">选择预设查看参数</p>}
            </div>
          </div>
          {confirmReplace && <p className="text-sm text-destructive" role="alert">应用「{selectedPreset?.name}」将替换当前自定义参数。</p>}
          <DialogFooter className="shrink-0 gap-2">
            <Button variant="ghost" onClick={() => {
              props.onSelectCustom();
              changeOpen(false);
            }}>自定义</Button>
            <Button variant="outline" onClick={() => changeOpen(false)}>取消</Button>
            <Button disabled={!selection || loading} onClick={apply}>{confirmReplace ? "替换并应用" : "应用预设"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
