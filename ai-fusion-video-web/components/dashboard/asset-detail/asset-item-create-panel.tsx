"use client";
import { AssetPropertiesEditor } from "./asset-properties-editor";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { assetApi } from "@/lib/api/asset";
import { Loader2, Plus, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { itemTypeOptions } from "./shared";
export function AssetItemCreatePanel({
  assetId,
  assetType,
  parentVersion,
  onClose,
  onCreated,
}: {
  assetId: number;
  assetType: string;
  parentVersion: number;
  onClose: () => void;
  onCreated: () => void;
}) {
  const requests = useCommandRequestKey();
  const [version] = useState(parentVersion);
  const [name, setName] = useState("");
  const [itemType, setItemType] = useState("");
  const [creating, setCreating] = useState(false);
  const [newItemProperties, setNewItemProperties] = useState<
    Record<string, unknown>
  >({});
  const handleNewItemPropertyChange = (key: string, value: string | null) => {
    setNewItemProperties((prev) => ({ ...prev, [key]: value }));
  };
  const handleCreate = async () => {
    if (!name.trim()) return;
    try {
      setCreating(true);
      const payload = {
        assetId,
        parentVersion: version,
        name: name.trim(),
        itemType: itemType || undefined,
        properties: newItemProperties,
      };
      await assetApi.createItem(payload, requests.forPayload(payload));
      requests.clear();
      onCreated();
    } catch (err) {
      console.error("创建子资产失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };
  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/30 bg-card/60 backdrop-blur-sm shrink-0">
        <Button variant="ghost" size="icon-sm" title="关闭" onClick={onClose}>
          <X className="h-3.5 w-3.5" />
        </Button>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold truncate">新建子资产</h3>
        </div>
        <Button
          variant="default"
          size="sm"
          title="保存"
          onClick={handleCreate}
          disabled={!name.trim() || creating}
        >
          {creating ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Plus className="h-3 w-3" />
          )}
          创建
        </Button>
      </div>

      {/* 创建表单 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-5 space-y-5">
        {/* 基本信息 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            基本信息
          </h4>
          <div className="rounded-xl border border-border/30 bg-card/40 p-3.5 space-y-3">
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">
                名称
              </label>
              <Input
                disabled={creating}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="子资产名称"
                className="h-8 text-xs"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreate();
                }}
                autoFocus
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">
                类型
              </label>
              <Select
                disabled={creating}
                value={itemType || ""}
                onValueChange={(v) => {
                  if (v) setItemType(v);
                }}
                items={itemTypeOptions}
              >
                <SelectTrigger size="sm" className="text-xs">
                  <SelectValue placeholder="选择类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {itemTypeOptions.map((opt) => (
                      <SelectItem
                        key={opt.value}
                        value={opt.value}
                        className="text-xs"
                      >
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        <AssetPropertiesEditor
          assetType={assetType}
          value={newItemProperties}
          onChange={handleNewItemPropertyChange}
          disabled={creating}
        />
      </div>
    </div>
  );
}
