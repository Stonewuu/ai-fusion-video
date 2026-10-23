"use client";
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
import { Textarea } from "@/components/ui/textarea";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { assetApi } from "@/lib/api/asset";
import { Loader2, Plus, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { assetTypeOptions, type CreateProps } from "./shared";
export function AssetCreatePanel(props: CreateProps) {
  const { onClose } = props;
  const createRequest = useCommandRequestKey();
  const [createName, setCreateName] = useState("");
  const [createType, setCreateType] = useState("character");
  const [createDescription, setCreateDescription] = useState("");
  const [createSaving, setCreateSaving] = useState(false);
  const handleCreate = async () => {
    if (!createName.trim()) return;
    try {
      setCreateSaving(true);
      const payload = {
        projectId: props.projectId,
        type: createType,
        name: createName.trim(),
        description: createDescription || null,
      };
      const created = await assetApi.create(
        payload,
        createRequest.forPayload(payload),
      );
      createRequest.clear();
      props.onCreated(created);
    } catch (err) {
      console.error("创建资产失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setCreateSaving(false);
    }
  };
  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border/20 shrink-0">
        <Button variant="ghost" size="icon-sm" title="关闭" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold">新建资产</h3>
        </div>
        <Button
          variant="default"
          size="sm"
          title="保存"
          onClick={handleCreate}
          disabled={!createName.trim() || createSaving}
        >
          {createSaving ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Plus className="h-3 w-3" />
          )}
          创建
        </Button>
      </div>

      {/* 表单 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-4 space-y-4">
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">名称</label>
          <Input
            disabled={createSaving}
            value={createName}
            onChange={(e) => setCreateName(e.target.value)}
            placeholder="资产名称"
            className="h-8 text-xs"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreate();
            }}
            autoFocus
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">类型</label>
          <Select
            disabled={createSaving}
            value={createType}
            onValueChange={(v) => {
              if (v) setCreateType(v);
            }}
            items={assetTypeOptions}
          >
            <SelectTrigger size="sm" className="text-xs">
              <SelectValue placeholder="选择类型" />
            </SelectTrigger>
            <SelectContent className="text-xs">
              <SelectGroup>
                {assetTypeOptions.map((t) => (
                  <SelectItem key={t.value} value={t.value} className="text-xs">
                    {t.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">描述</label>
          <Textarea
            disabled={createSaving}
            value={createDescription}
            onChange={(e) => setCreateDescription(e.target.value)}
            placeholder="资产描述（可选）..."
            rows={3}
            className="resize-none text-xs"
          />
        </div>
      </div>
    </div>
  );
}
