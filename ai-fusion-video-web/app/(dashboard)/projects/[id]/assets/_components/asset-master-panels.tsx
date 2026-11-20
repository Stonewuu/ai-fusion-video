"use client";

import * as React from "react";
import { Edit3, Trash2 } from "lucide-react";
import type { AssetWithItems } from "@/lib/api/asset";
import { Button } from "@/components/ui/button";
import { Dialog, DialogTrigger } from "@/components/ui/dialog";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { AssetMasterFields, AssetMasterFeedback } from "./asset-master-fields";
import { AssetMasterDialogContent } from "./asset-master-dialog";
import { getAssetType } from "./asset-types";

interface MasterAssetHeaderProps {
  asset: AssetWithItems;
  saving: boolean;
  disabled: boolean;
  onEditingChange: (open: boolean) => void;
  onSave: (
    base: AssetWithItems,
    payload: { name: string; type: string; description: string },
  ) => Promise<void>;
  onDelete: (base: AssetWithItems) => void;
}

export function MasterAssetHeader({
  asset,
  saving,
  disabled,
  onEditingChange,
  onSave,
  onDelete,
}: MasterAssetHeaderProps) {
  const [error, setError] = React.useState("");
  const [base, setBase] = React.useState(asset);
  const [editing, setEditing] = React.useState(false);
  const [name, setName] = React.useState(asset.name);
  const [type, setType] = React.useState(asset.type);
  const [description, setDescription] = React.useState(asset.description || "");
  const [nameTouched, setNameTouched] = React.useState(false);
  const nameRef = React.useRef<HTMLInputElement>(null);
  const assetType = getAssetType(asset.type);
  const TypeIcon = assetType.icon;
  const blocked = saving || disabled;
  const dirty =
    name !== base.name ||
    type !== base.type ||
    description !== (base.description || "");

  const resetFields = () => {
    setBase(asset);
    setName(asset.name);
    setType(asset.type);
    setDescription(asset.description || "");
    setNameTouched(false);
    setError("");
  };
  const save = async () => {
    setNameTouched(true);
    if (!name.trim()) {
      nameRef.current?.focus();
      return;
    }
    if (!dirty) {
      return;
    }
    try {
      setError("");
      await onSave(base, { name, type, description });
      setEditing(false);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };

  return (
    <Dialog
      open={editing}
      onOpenChange={(open, details) => {
        if (blocked) {
          details.cancel();
          return;
        }
        if (open) {
          resetFields();
          onEditingChange(true);
        }
        setEditing(open);
      }}
      onOpenChangeComplete={(open) => {
        if (!open) {
          onEditingChange(false);
        }
      }}
    >
      <section className="border-b border-border/20 bg-muted/20 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">
                主资产
              </span>
              <span
                className={
                  "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium " +
                  assetType.bg +
                  " " +
                  assetType.color
                }
              >
                <TypeIcon className="size-3" />
                {assetType.label}
              </span>
            </div>
            <h2 className="mt-1 truncate text-xl font-semibold tracking-tight">
              {asset.name}
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              {asset.items.length} 个子资产
            </p>
          </div>
          <div className="flex items-center gap-2">
            <DialogTrigger
              disabled={blocked}
              aria-label="编辑主资产"
              render={<Button variant="outline" />}
            >
              <Edit3 />
              编辑
            </DialogTrigger>
            <Button
              variant="destructive-ghost"
              size="icon"
              disabled={blocked}
              onClick={() => onDelete(asset)}
              title="删除主资产"
              aria-label="删除主资产"
            >
              <Trash2 />
            </Button>
          </div>
        </div>
        <div className="mt-4 rounded-lg border border-border/20 bg-background/70 p-3">
          <p className="text-xs font-medium text-muted-foreground">资产设定</p>
          <p className="mt-1 line-clamp-2 text-sm leading-5 text-foreground/80">
            {asset.description || "—"}
          </p>
        </div>
      </section>
      <AssetMasterDialogContent
        mode="edit"
        description={asset.name}
        saving={saving}
        disabled={disabled || !editing}
        canSubmit={Boolean(name.trim()) && dirty}
        nameRef={nameRef}
        onCancel={() => setEditing(false)}
        onSubmit={save}
      >
        <AssetMasterFields
          name={name}
          type={type}
          description={description}
          disabled={blocked || !editing}
          showNameError={nameTouched && !name.trim()}
          nameRef={nameRef}
          onName={(value) => {
            setName(value);
            setNameTouched(true);
          }}
          onNameBlur={() => setNameTouched(true)}
          onType={setType}
          onDescription={setDescription}
        />
        <AssetMasterFeedback message={error} />
      </AssetMasterDialogContent>
    </Dialog>
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
  const [open, setOpen] = React.useState(true);
  const [error, setError] = React.useState("");
  const [name, setName] = React.useState("");
  const [type, setType] = React.useState("character");
  const [description, setDescription] = React.useState("");
  const [nameTouched, setNameTouched] = React.useState(false);
  const nameRef = React.useRef<HTMLInputElement>(null);

  const create = async () => {
    setNameTouched(true);
    if (!name.trim()) {
      nameRef.current?.focus();
      return;
    }
    setError("");
    try {
      await onSubmit(name, type, description);
      setOpen(false);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen, details) => {
        if (saving) {
          details.cancel();
          return;
        }
        setOpen(nextOpen);
      }}
      onOpenChangeComplete={(nextOpen) => {
        if (!nextOpen) {
          onCancel();
        }
      }}
    >
      <AssetMasterDialogContent
        mode="create"
        description="添加角色、场景或道具。"
        saving={saving}
        disabled={!open}
        canSubmit={Boolean(name.trim())}
        nameRef={nameRef}
        onCancel={() => setOpen(false)}
        onSubmit={create}
      >
        <AssetMasterFields
          name={name}
          type={type}
          description={description}
          disabled={saving || !open}
          showNameError={nameTouched && !name.trim()}
          nameRef={nameRef}
          onName={(value) => {
            setName(value);
            setNameTouched(true);
          }}
          onNameBlur={() => setNameTouched(true)}
          onType={setType}
          onDescription={setDescription}
        />
        <AssetMasterFeedback message={error} />
      </AssetMasterDialogContent>
    </Dialog>
  );
}
