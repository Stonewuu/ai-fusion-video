"use client";

import { CheckCircle2, FileText, FileUp, Loader2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const encodings = [
  { value: "utf-8", label: "UTF-8" },
  { value: "gb18030", label: "GB18030 / GBK" },
  { value: "utf-16le", label: "UTF-16 LE" },
  { value: "utf-16be", label: "UTF-16 BE" },
];

interface Props {
  file: File | null;
  source: string;
  encoding: string;
  reading: boolean;
  disabled: boolean;
  onChoose: () => void;
  onRemove: () => void;
  onEncoding: (value: string) => void;
}

export function SourceInputFile({
  file,
  source,
  encoding,
  reading,
  disabled,
  onChoose,
  onRemove,
  onEncoding,
}: Props) {
  if (!file) {
    return (
      <div className="flex min-h-72 flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-input bg-muted/20 px-4 py-8 text-center">
        <div className="flex size-14 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <FileUp className="size-7" aria-hidden="true" />
        </div>
        <div className="flex flex-col gap-2">
          <h3 className="text-base font-semibold">导入小说或剧本</h3>
          <p className="text-sm text-muted-foreground">
            上传原文，自动整理为分集目录
          </p>
        </div>
        <Button variant="outline" disabled={disabled} onClick={onChoose}>
          <Upload />
          选择文件
        </Button>
        <p className="text-xs text-muted-foreground">
          TXT / Markdown · 最多 100 MB、2000 万字
        </p>
        {source && (
          <p className="text-xs text-muted-foreground">
            已粘贴 {source.length.toLocaleString()} 字，选择文件后将替换原文。
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <FileText className="size-5" aria-hidden="true" />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <p className="truncate text-sm font-medium" title={file.name}>
            {file.name}
          </p>
          <p
            className="flex items-center gap-1.5 text-xs text-muted-foreground"
            role="status"
          >
            {reading ? (
              <Loader2 className="size-3 animate-spin" />
            ) : source ? (
              <CheckCircle2 className="size-3 text-primary" />
            ) : null}
            {reading
              ? "正在读取文件"
              : source
                ? `已读取 ${source.length.toLocaleString()} 字`
                : "文件读取失败"}
          </p>
        </div>
        <Button
          variant="ghost"
          disabled={disabled || reading}
          onClick={onRemove}
        >
          移除
        </Button>
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <span>原文预览</span>
          {source.length > 1200 && <span>前 1,200 字</span>}
        </div>
        <div
          aria-label="文件原文预览"
          tabIndex={0}
          className="h-52 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-border/20 bg-muted/20 p-4 text-sm leading-7 outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
        >
          {reading ? (
            <span className="text-muted-foreground">正在读取文件…</span>
          ) : source ? (
            source.slice(0, 1200)
          ) : (
            <span className="text-muted-foreground">
              请检查文件编码，或重新选择文件。
            </span>
          )}
        </div>
      </div>
      <div className="grid min-w-0 gap-2 sm:grid-cols-[minmax(0,12rem)_minmax(0,1fr)] sm:items-end">
        <div className="flex min-w-0 flex-col gap-2">
          <label className="text-sm font-medium" htmlFor="source-encoding">
            文件编码
          </label>
          <Select
            items={encodings}
            value={encoding}
            disabled={disabled || reading}
            onValueChange={(value) => {
              if (value) {
                onEncoding(value);
              }
            }}
          >
            <SelectTrigger id="source-encoding" className="w-full">
              <SelectValue placeholder="选择编码" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {encodings.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <p className="text-xs leading-5 text-muted-foreground sm:pb-2">
          预览出现乱码时，切换文件编码。
        </p>
      </div>
    </div>
  );
}
