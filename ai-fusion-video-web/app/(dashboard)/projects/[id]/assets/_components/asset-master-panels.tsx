"use client";

import * as React from "react";
import { Check, Edit3, Loader2, Plus, Trash2, X } from "lucide-react";
import type { AssetWithItems } from "@/lib/api/asset";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ASSET_TYPES, getAssetType } from "./asset-types";

interface MasterAssetHeaderProps {
  asset: AssetWithItems;
  saving: boolean;
  onSave: (payload: {
    name: string;
    type: string;
    description: string;
  }) => Promise<void>;
  onDelete: () => void;
}

export function MasterAssetHeader({
  asset,
  saving,
  onSave,
  onDelete,
}: MasterAssetHeaderProps) {
  const [editing, setEditing] = React.useState(false);
  const [name, setName] = React.useState(asset.name);
  const [type, setType] = React.useState(asset.type);
  const [description, setDescription] = React.useState(asset.description || "");
  const assetType = getAssetType(asset.type);
  const TypeIcon = assetType.icon;

  const cancel = () => {
    setName(asset.name);
    setType(asset.type);
    setDescription(asset.description || "");
    setEditing(false);
  };

  const save = async () => {
    await onSave({ name, type, description });
    setEditing(false);
  };

  return (
    <section className="border-b border-border/40 bg-gradient-to-br from-primary/[0.08] via-card/85 to-card/60 px-5 py-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">
              主资产
            </span>
            <span
              className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium ${assetType.bg} ${assetType.color}`}
            >
              <TypeIcon className="h-3 w-3" />
              {assetType.label}
            </span>
          </div>
          <h2 className="mt-1 truncate text-xl font-semibold tracking-tight">
            {asset.name}
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {asset.items?.length || 0} 个子资产
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setEditing((value) => !value)}
          >
            <Edit3 className="h-3.5 w-3.5" />
            {editing ? "收起" : "编辑"}
          </Button>
          <Button
            variant="destructive-ghost"
            size="icon-sm"
            onClick={onDelete}
            title="删除主资产"
            aria-label="删除主资产"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {editing ? (
        <div className="mt-4 rounded-2xl border border-border/50 bg-background/55 p-3.5">
          <div className="grid gap-2.5 sm:grid-cols-[minmax(0,1fr)_140px]">
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="资产名称"
              className="h-9 bg-background/80"
            />
            <Select
              value={type}
              onValueChange={(value) => setType(String(value))}
              items={ASSET_TYPES.map((item) => ({
                value: item.value,
                label: item.label,
              }))}
            >
              <SelectTrigger className="h-9 bg-background/80">
                <SelectValue placeholder="资产类型" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {ASSET_TYPES.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
          <Textarea
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            rows={3}
            placeholder="资产设定"
            className="mt-2.5 resize-none bg-background/80 text-sm leading-5"
          />
          <div className="mt-2.5 flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={cancel}>
              <X className="h-3.5 w-3.5" />
              取消
            </Button>
            <Button size="sm" disabled={saving || !name.trim()} onClick={() => void save()}>
              {saving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Check className="h-3.5 w-3.5" />
              )}
              保存
            </Button>
          </div>
        </div>
      ) : (
        <div className="mt-4 rounded-2xl border border-border/40 bg-background/45 px-3.5 py-3">
          <p className="text-[10px] font-medium text-muted-foreground">资产设定</p>
          <p className="mt-1 line-clamp-2 text-sm leading-5 text-foreground/80">
            {asset.description || "未填写"}
          </p>
        </div>
      )}
    </section>
  );
}

interface MasterAssetFormProps {
  saving: boolean;
  onCancel: () => void;
  onSubmit: (name: string, type: string, description: string) => Promise<void>;
}

export function MasterAssetForm({
  saving,
  onCancel,
  onSubmit,
}: MasterAssetFormProps) {
  const [name, setName] = React.useState("");
  const [type, setType] = React.useState("character");
  const [description, setDescription] = React.useState("");

  return (
    <div className="mx-auto w-full max-w-2xl p-7">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">
            新建
          </p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight">主资产</h2>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onCancel} title="取消" aria-label="取消">
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="mt-6 rounded-2xl border border-border/50 bg-background/55 p-4">
        <div className="grid gap-2.5 sm:grid-cols-[minmax(0,1fr)_140px]">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="资产名称"
            className="h-9 bg-background/80"
            autoFocus
          />
          <Select
            value={type}
            onValueChange={(value) => setType(String(value))}
            items={ASSET_TYPES.map((item) => ({
              value: item.value,
              label: item.label,
            }))}
          >
            <SelectTrigger className="h-9 bg-background/80">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {ASSET_TYPES.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <Textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          rows={4}
          placeholder="资产设定"
          className="mt-2.5 resize-none bg-background/80 text-sm leading-5"
        />
        <div className="mt-3 flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={onCancel}>
            取消
          </Button>
          <Button
            size="sm"
            disabled={saving || !name.trim()}
            onClick={() => void onSubmit(name, type, description)}
          >
            {saving ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Plus className="h-3.5 w-3.5" />
            )}
            创建
          </Button>
        </div>
      </div>
    </div>
  );
}
