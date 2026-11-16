"use client";

import { useId, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { uploadGenerationReference } from "@/components/dashboard/generation/generation-reference-upload";
import type { ComfyUiBindingValueType } from "@/lib/api/comfyui-workflow";

export interface ComfyUiTestField {
  name: string;
  type: ComfyUiBindingValueType;
  optional?: boolean;
}
const labels: Record<string, string> = {
  prompt: "提示词",
  negativePrompt: "负面提示词",
  width: "宽度",
  height: "高度",
  seed: "随机种子",
  count: "数量",
  duration: "时长",
  referenceImages: "参考图片",
  firstFrame: "首帧",
  lastFrame: "尾帧",
  referenceVideos: "参考视频",
  referenceAudios: "参考音频",
  referenceImageUrls: "参考图片",
  firstFrameUrl: "首帧",
  lastFrameUrl: "尾帧",
};

const uploadSettings = {
  image: { label: "图片", accept: "image/*", maxMb: 20 },
  video: { label: "视频", accept: "video/*", maxMb: 100 },
  audio: { label: "音频", accept: "audio/*", maxMb: 100 },
} as const;
type UploadKind = keyof typeof uploadSettings;

export function ComfyUiTestFields({
  fields,
  values,
  disabled,
  onChange,
  onAppend,
  onUploading,
}: {
  fields: ComfyUiTestField[];
  values: Record<string, string>;
  disabled: boolean;
  onChange: (field: string, value: string) => void;
  onAppend: (field: string, references: string[]) => void;
  onUploading: (field: string, uploading: boolean) => void;
}) {
  return (
    <div className="grid min-w-0 gap-4 md:grid-cols-2">
      {fields.map((field) => (
        <TestField
          key={field.name}
          field={field}
          value={values[field.name] ?? ""}
          disabled={disabled}
          onChange={(value) => onChange(field.name, value)}
          onAppend={(refs) => onAppend(field.name, refs)}
          onUploading={(busy) => onUploading(field.name, busy)}
        />
      ))}
    </div>
  );
}

function TestField({
  field,
  value,
  disabled,
  onChange,
  onAppend,
  onUploading,
}: {
  field: ComfyUiTestField;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onAppend: (refs: string[]) => void;
  onUploading: (uploading: boolean) => void;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const list =
    field.type === "string_list" || field.type.startsWith("uploaded_");
  const label = labels[field.name] ?? field.name;
  const mediaKind =
    field.type === "uploaded_image"
      ? "image"
      : field.type === "uploaded_video"
        ? "video"
        : field.type === "uploaded_audio"
          ? "audio"
          : null;
  const upload = async (files: File[], kind: UploadKind) => {
    const config = uploadSettings[kind];
    setUploading(true);
    onUploading(true);
    setError(null);
    try {
      if (
        files.length > 64 ||
        files.some(
          (file) =>
            !file.type.startsWith(`${kind}/`) ||
            file.size > config.maxMb * 1024 * 1024,
        )
      ) {
        throw new Error(
          `每次最多上传 64 个不超过 ${config.maxMb}MB 的${config.label}文件。`,
        );
      }
      for (const file of files) {
        onAppend([await uploadGenerationReference(file, kind)]);
      }
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setUploading(false);
      onUploading(false);
    }
  };
  return (
    <div className="flex min-w-0 flex-col gap-2 p-1">
      <Label htmlFor={id}>
        {label}
        {field.optional && (
          <span className="text-xs text-muted-foreground">（可选）</span>
        )}
      </Label>
      {list ? (
        <Textarea
          id={id}
          value={value}
          disabled={disabled || uploading}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? `${id}-error` : undefined}
          onChange={(event) => onChange(event.target.value)}
          placeholder="每行一项，可填写媒体引用、URL 或 Data URI"
        />
      ) : (
        <Input
          id={id}
          type={
            field.type === "integer" || field.type === "number"
              ? "number"
              : "text"
          }
          step={field.type === "integer" ? "1" : "any"}
          value={value}
          disabled={disabled}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? `${id}-error` : undefined}
          placeholder={
            field.type === "boolean"
              ? "true 或 false"
              : "留空使用工作流中保存的值"
          }
          onChange={(event) => onChange(event.target.value)}
        />
      )}
      {field.optional && (
        <p className="text-xs text-muted-foreground">留空时不使用此输入</p>
      )}
      {mediaKind && (
        <>
          <input
            ref={input}
            type="file"
            multiple
            accept={uploadSettings[mediaKind].accept}
            className="hidden"
            aria-label={`上传${label}`}
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              event.target.value = "";
              void upload(files, mediaKind);
            }}
          />
          <Button
            variant="outline"
            size="sm"
            disabled={disabled || uploading}
            onClick={() => input.current?.click()}
          >
            <Upload />
            {uploading ? "上传中…" : `上传${uploadSettings[mediaKind].label}`}
          </Button>
        </>
      )}
      {error && (
        <p id={`${id}-error`} role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
