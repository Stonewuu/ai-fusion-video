"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

import { useId, useState } from "react";
import { ImagePlus, Link2, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { SafeImage } from "@/components/media/managed-image";
import { mediaId } from "@/lib/api/media";
import type { DialogueElement } from "@/lib/api/script";
import type { ResourceField } from "@/lib/canvas/resource-fields";
import type { CanvasLibraryEntry } from "@/lib/canvas/resource-library";
import { CanvasChoiceField, CanvasNumberField } from "./field-controls";
import { CanvasTagsField } from "./tags-field";
import { CanvasStructuredDocument } from "./structured-value-editor";
import { resourceFieldSuggestions } from "@/lib/canvas/resource-field-presentation";
import {
  CanvasCharactersEditor,
  CanvasDialogueEditor,
  type CanvasCharacter,
} from "./structured-text";
import { CanvasMediaPicker } from "../library/media-picker";

export function CanvasResourceFieldControl({
  field,
  value,
  onChange,
  disabled = false,
  library = [],
}: {
  field: ResourceField;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
  library?: CanvasLibraryEntry[];
}) {
  const id = useId();
  const [picking, setPicking] = useState(false);
  const [search, setSearch] = useState("");
  const media = ["image", "video", "audio"].includes(field.kind);
  const relation = ["asset", "assetItem", "scriptEpisode"].includes(field.kind);
  const selected = field.multiple
    ? Array.isArray(value)
      ? (value as number[])
      : []
    : typeof value === "number"
      ? [value]
      : [];
  const related = library.filter(
    (item) =>
      item.ref.type === field.kind &&
      (!field.assetType || item.assetType === field.assetType),
  );
  const options = field.options ? [...field.options] : [];
  if (
    field.kind === "choice" &&
    value !== null &&
    value !== undefined &&
    !options.some((option) => option.value === String(value))
  )
    options.push({ value: String(value), label: `当前值：${String(value)}` });
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium">
          {field.label}
          {field.required ? " *" : ""}
        </label>
        {!field.required &&
          value !== null &&
          value !== undefined &&
          value !== "" &&
          !disabled && (
            <Button
              size="xs"
              variant="ghost"
              title={`清空${field.label}`}
              aria-label={`清空${field.label}`}
              onClick={() =>
                onChange(
                  field.multiple ||
                    ["dialogues", "characters", "tags"].includes(field.kind)
                    ? []
                    : null,
                )
              }
            >
              清空
            </Button>
          )}
      </div>
      {media ? (
        <div className="space-y-2">
          {typeof value === "string" && value && (
            <div className="flex items-center gap-2 rounded-lg border border-border/20 bg-background/70 p-2">
              {field.kind === "image" && (
                <SafeImage
                  src={value}
                  alt={field.label}
                  className="size-16 shrink-0 rounded-lg object-cover"
                />
              )}
              <p className="min-w-0 truncate text-xs text-muted-foreground">
                已关联{field.label}
              </p>
            </div>
          )}
          <Button
            id={id}
            aria-label={`选择${field.label}`}
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={() => setPicking(true)}
          >
            <ImagePlus />
            选择素材
          </Button>
          <Dialog open={picking} onOpenChange={setPicking}>
            <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
              <DialogHeader className="shrink-0">
                <DialogTitle>选择{field.label}</DialogTitle>
                <DialogDescription>
                  选择{field.kind === "image" ? "图片" : field.kind === "video" ? "视频" : "音频"}，也可以上传新文件。
                </DialogDescription>
              </DialogHeader>
              <DialogScrollBody>
                <CanvasMediaPicker
                  kind={
                    field.kind === "image"
                      ? "IMAGE"
                      : field.kind === "video"
                        ? "VIDEO"
                        : "AUDIO"
                  }
                  value={
                    typeof value === "string"
                      ? (mediaId(value) ?? undefined)
                      : undefined
                  }
                  onSelect={(item) => {
                    onChange(`media://${item.id}`);
                    setPicking(false);
                  }}
                />
              </DialogScrollBody>
            </DialogContent>
          </Dialog>
        </div>
      ) : relation ? (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {selected.map((selectedId) => (
              <span
                key={selectedId}
                className="inline-flex max-w-full items-center gap-1 rounded-lg border border-border/20 px-2 py-1 text-xs"
              >
                <span className="truncate">
                  {related.find((item) => item.ref.id === selectedId)?.title ??
                    "关联内容暂不可读取"}
                </span>
                {!disabled && (
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    title="移除关联"
                    onClick={() =>
                      onChange(
                        field.multiple
                          ? selected.filter((item) => item !== selectedId)
                          : null,
                      )
                    }
                  >
                    <X />
                  </Button>
                )}
              </span>
            ))}
          </div>
          <Button
            id={id}
            aria-label={`选择${field.label}`}
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={() => setPicking(true)}
          >
            <Link2 />
            选择关联
          </Button>
          <Dialog open={picking} onOpenChange={setPicking}>
            <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
              <DialogHeader className="shrink-0">
                <DialogTitle>{field.label}</DialogTitle>
                <DialogDescription>
                  仅显示当前项目中可关联的内容。
                </DialogDescription>
              </DialogHeader>
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                aria-label="搜索关联内容"
                placeholder="搜索"
              />
              <DialogScrollBody className="space-y-3">
                {related
                  .filter((item) =>
                    item.title.toLowerCase().includes(search.toLowerCase()),
                  )
                  .map((item) => (
                    <label
                      key={item.ref.id}
                      className="flex items-center gap-3 rounded-lg p-2 hover:bg-muted/50"
                    >
                      <Checkbox
                        checked={selected.includes(item.ref.id)}
                        onCheckedChange={(checked) =>
                          onChange(
                            field.multiple
                              ? checked
                                ? [...selected, item.ref.id]
                                : selected.filter(
                                    (selectedId) => selectedId !== item.ref.id,
                                  )
                              : checked
                                ? item.ref.id
                                : null,
                          )
                        }
                      />
                      <span className="text-sm">{item.title}</span>
                    </label>
                  ))}
              </DialogScrollBody>
              <DialogFooter className="shrink-0">
                <Button onClick={() => setPicking(false)}>完成</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      ) : field.kind === "dialogues" ? (
        <CanvasDialogueEditor
          value={Array.isArray(value) ? (value as DialogueElement[]) : []}
          onChange={onChange}
          disabled={disabled}
        />
      ) : field.kind === "characters" ? (
        <CanvasCharactersEditor
          value={Array.isArray(value) ? (value as CanvasCharacter[]) : []}
          onChange={onChange}
          disabled={disabled}
        />
      ) : field.kind === "tags" ? (
        <CanvasTagsField label={field.label} value={Array.isArray(value) ? value as string[] : []} onChange={onChange} disabled={disabled} />
      ) : field.kind === "json" ? (
        <CanvasStructuredDocument label={field.label} value={value} jsonType={field.jsonType} onChange={onChange} disabled={disabled} />
      ) : field.kind === "choice" ? (
        <CanvasChoiceField
          value={value === null || value === undefined ? null : String(value)}
          items={options}
          label={field.label}
          placeholder={`选择${field.label}`}
          disabled={disabled}
          onChange={(next) =>
            onChange(field.key === "status" ? Number(next) : next)
          }
        />
      ) : field.kind === "number" ? (
        <CanvasNumberField
          id={id}
          label={field.label}
          value={typeof value === "number" ? value : undefined}
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          disabled={disabled}
          placeholder="填写数值"
          unit={field.key.toLowerCase().includes("duration") ? "秒" : undefined}
          onChange={(next) => onChange(next ?? null)}
        />
      ) : field.kind === "long" ? (
        <Textarea
          id={id}
          rows={field.key === "rawContent" || field.key === "content" ? 10 : 4}
          value={typeof value === "string" ? value : ""}
          placeholder={`填写${field.label}…`}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
        />
      ) : resourceFieldSuggestions[field.key] ? (
        <CanvasChoiceField label={field.label} value={value == null ? null : String(value)}
          items={resourceFieldSuggestions[field.key].map((item) => ({ value: item, label: item }))}
          custom disabled={disabled} placeholder={`选择${field.label}`} onChange={onChange} />
      ) : (
        <Input
          id={id}
          value={value === null || value === undefined ? "" : String(value)}
          placeholder={`填写${field.label}`}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
        />
      )}
    </div>
  );
}
