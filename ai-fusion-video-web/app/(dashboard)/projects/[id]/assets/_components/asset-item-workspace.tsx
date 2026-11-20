"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Loader2, Plus, RotateCcw } from "lucide-react";
import type { AssetItem } from "@/lib/api/asset";
import { uploadFile } from "@/lib/api/storage";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { getAppearance, getAppearanceHints } from "./asset-types";
import { AssetItemImage } from "./asset-item-image";
import { AssetItemFields } from "./asset-item-fields";
import { AssetItemMessage, AssetItemSaveStatus } from "./asset-item-feedback";

interface AssetItemWorkspaceProps {
  mode: "create" | "edit";
  initialSection: "details" | "image";
  item: AssetItem | null;
  assetType: string;
  parentVersion: number;
  saving: boolean;
  onUploadingChange: (uploading: boolean) => void;
  onCreate: (
    parentVersion: number,
    name: string,
    appearance: string,
    imageUrl: string,
    generateAfterCreate: boolean,
  ) => Promise<void>;
  onSave: (
    item: AssetItem,
    name: string,
    appearance: string,
    imageUrl: string,
  ) => Promise<AssetItem>;
  onClose: () => void;
  onGenerate: () => void;
}

export function AssetItemWorkspace({
  mode,
  initialSection,
  item,
  assetType,
  parentVersion,
  saving,
  onUploadingChange,
  onCreate,
  onSave,
  onClose,
  onGenerate,
}: AssetItemWorkspaceProps) {
  const [open, setOpen] = useState(true);
  const [base, setBase] = useState(item);
  const [creationVersion] = useState(parentVersion);
  const initialName = base?.name || "";
  const initialAppearance = getAppearance(base);
  const initialImageUrl = base?.imageUrl || "";
  const [name, setName] = useState(initialName);
  const [appearance, setAppearance] = useState(initialAppearance);
  const [imageUrl, setImageUrl] = useState(initialImageUrl);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const uploadButtonRef = useRef<HTMLButtonElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const appearanceHints = useMemo(
    () => getAppearanceHints(assetType),
    [assetType],
  );
  const dirty =
    name !== initialName ||
    appearance !== initialAppearance ||
    imageUrl !== initialImageUrl;
  const disabled = saving || uploading || !open || !name.trim();
  const invalidName = !name.trim() && (nameTouched || mode === "edit");

  useEffect(() => {
    if (
      !item ||
      !base ||
      dirty ||
      saving ||
      uploading ||
      (item.version <= base.version && item.assetVersion <= base.assetVersion)
    ) {
      return;
    }
    setBase(item);
    setName(item.name || "");
    setAppearance(getAppearance(item));
    setImageUrl(item.imageUrl || "");
  }, [item, base, dirty, saving, uploading]);

  const upload = async (file: File) => {
    if (saving || uploading) {
      return;
    }
    if (!file.type.startsWith("image/")) {
      setUploadError("请选择图片文件");
      return;
    }

    setUploading(true);
    onUploadingChange(true);
    setUploadError("");
    try {
      setImageUrl(await uploadFile(file, "assets"));
    } catch (cause) {
      setUploadError(getApiErrorMessage(cause));
    } finally {
      setUploading(false);
      onUploadingChange(false);
    }
  };

  const appendHint = (hint: string) => {
    const token = `${hint}：`;
    setAppearance((current) => {
      if (current.includes(token)) {
        return current;
      }
      return current.trim() ? `${current.trim()}\n${token}` : token;
    });
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const save = async () => {
    if (!base || disabled) {
      return;
    }
    setError("");
    const updated = await onSave(base, name, appearance, imageUrl);
    setBase(updated);
    setName(updated.name || "");
    setAppearance(getAppearance(updated));
    setImageUrl(updated.imageUrl || "");
  };

  const create = async (generateAfterCreate: boolean) => {
    if (disabled) {
      return;
    }
    setError("");
    await onCreate(
      creationVersion,
      name,
      appearance,
      imageUrl,
      generateAfterCreate,
    );
  };

  const generate = async () => {
    if (mode === "create") {
      await create(true);
      setOpen(false);
      return;
    }
    if (base && dirty) {
      await save();
    }
    onGenerate();
    setOpen(false);
  };

  const reset = () => {
    setName(initialName);
    setAppearance(initialAppearance);
    setImageUrl(initialImageUrl);
    setError("");
    setUploadError("");
    setNameTouched(false);
    nameRef.current?.focus({ preventScroll: true });
  };

  const submit = async () => {
    if (disabled || (mode === "edit" && !dirty)) {
      return;
    }
    try {
      if (mode === "create") {
        await create(false);
      } else {
        await save();
      }
      setOpen(false);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!saving && !uploading) {
          setOpen(nextOpen);
        }
      }}
      onOpenChangeComplete={(nextOpen) => {
        if (!nextOpen) {
          onClose();
        }
      }}
    >
      <DialogContent
        initialFocus={initialSection === "image" ? uploadButtonRef : nameRef}
        showCloseButton={!saving && !uploading}
        style={{ backgroundColor: "var(--background)" }}
        className="flex w-[calc(100%-2rem)] max-h-[calc(100dvh-2rem)] flex-col gap-0 overflow-hidden rounded-2xl p-0 duration-200 data-closed:duration-150 motion-reduce:animate-none sm:max-w-5xl"
      >
        <DialogHeader className="shrink-0 border-b border-border/20 p-5 pr-14">
          <DialogTitle className="text-lg font-semibold">
            {mode === "create" ? "新增子资产" : "编辑子资产"}
          </DialogTitle>
          <DialogDescription>
            {mode === "create" ? "设置名称、外观与资产图片。" : initialName}
          </DialogDescription>
        </DialogHeader>
        <form
          className="flex min-h-0 min-w-0 flex-col"
          noValidate
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <DialogScrollBody className="p-0">
            <div className="grid min-w-0 md:grid-cols-2">
              <AssetItemFields
                name={name}
                appearance={appearance}
                disabled={saving || uploading}
                invalidName={invalidName}
                nameRef={nameRef}
                textareaRef={textareaRef}
                hints={appearanceHints}
                onName={(value) => {
                  setName(value);
                  setNameTouched(true);
                }}
                onNameBlur={() => setNameTouched(true)}
                onAppearance={setAppearance}
                onHint={appendHint}
              />
              <div className="min-w-0 md:-order-1">
                <AssetItemImage
                  uploadButtonRef={uploadButtonRef}
                  imageUrl={imageUrl}
                  name={name}
                  assetType={assetType}
                  uploading={uploading}
                  saving={saving}
                  generationDisabled={disabled}
                  generationLabel={
                    mode === "create"
                      ? "创建并生图"
                      : dirty
                        ? "保存并生图"
                        : "生成图片"
                  }
                  error={uploadError}
                  onUpload={(file) => void upload(file)}
                  onRemove={() => {
                    setImageUrl("");
                    setUploadError("");
                  }}
                  onGenerate={() =>
                    void generate().catch((failure) =>
                      setError(getApiErrorMessage(failure)),
                    )
                  }
                />
              </div>
            </div>
          </DialogScrollBody>
          <div className="shrink-0 border-t border-border/20 bg-muted/20 px-5 py-4">
            <AssetItemMessage message={error} />
            <DialogFooter className="flex-row flex-wrap items-center justify-between gap-3 sm:justify-between">
              <AssetItemSaveStatus
                creating={mode === "create"}
                dirty={dirty}
                saving={saving}
              />
              <div className="flex items-center gap-2">
                {mode === "edit" && (
                  <Button
                    type="button"
                    variant="ghost"
                    disabled={saving || uploading || !dirty}
                    onClick={reset}
                  >
                    <RotateCcw />
                    还原
                  </Button>
                )}
                <Button
                  type="button"
                  variant="outline"
                  disabled={saving || uploading}
                  onClick={() => setOpen(false)}
                >
                  取消
                </Button>
                <Button
                  type="submit"
                  className="min-w-20"
                  disabled={disabled || (mode === "edit" && !dirty)}
                >
                  {saving ? (
                    <Loader2 className="animate-spin" />
                  ) : mode === "create" ? (
                    <Plus />
                  ) : (
                    <Check />
                  )}
                  {saving ? "正在保存" : mode === "create" ? "创建" : "保存"}
                </Button>
              </div>
            </DialogFooter>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
