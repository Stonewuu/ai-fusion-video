"use client";

import { useRef, useState } from "react";
import { FileText, Loader2, Upload } from "lucide-react";
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
import type { ScriptSplitRule } from "@/lib/api/script";
import {
  MAX_SOURCE_LENGTH,
  MAX_EPISODE_LENGTH,
  readSourceFile,
} from "@/lib/scripts/source-import";

const rules = [
  { value: "AUTO", label: "自动识别" },
  { value: "CHAPTER", label: "按章节标题" },
  { value: "EPISODE", label: "按分集标题" },
  { value: "LENGTH", label: "按字数分集" },
];
const encodings = [
  { value: "utf-8", label: "UTF-8" },
  { value: "gb18030", label: "GB18030 / GBK" },
  { value: "utf-16le", label: "UTF-16 LE" },
  { value: "utf-16be", label: "UTF-16 BE" },
];

interface Props {
  file: File | null;
  encoding: string;
  onFile: (file: File | null) => void;
  onEncoding: (encoding: string) => void;
  source: string;
  rule: ScriptSplitRule;
  targetLength: number;
  disabled: boolean;
  hasLegacySource: boolean;
  onSource: (source: string) => void;
  onRule: (rule: ScriptSplitRule) => void;
  onTargetLength: (length: number) => void;
  onError: (error: string) => void;
  onLegacySource: () => void;
}

export function SourceInput({
  file,
  encoding,
  onFile: setFile,
  onEncoding: setEncoding,
  source,
  rule,
  targetLength,
  disabled,
  hasLegacySource,
  onSource,
  onRule,
  onTargetLength,
  onError,
  onLegacySource,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [reading, setReading] = useState(false);

  const read = async (nextFile: File, nextEncoding: string) => {
    setFile(nextFile);
    setReading(true);
    onSource("");
    onError("");
    try {
      onSource(await readSourceFile(nextFile, nextEncoding));
    } catch (error) {
      onError(error instanceof Error ? error.message : "文件读取失败，请重试");
    } finally {
      setReading(false);
    }
  };

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={fileRef}
          type="file"
          accept=".txt,.md,.markdown"
          className="hidden"
          aria-label="选择原文文件"
          onChange={(event) => {
            const nextFile = event.target.files?.[0];
            if (nextFile) {
              void read(nextFile, encoding);
            }
            event.target.value = "";
          }}
        />
        <Button
          variant="outline"
          disabled={disabled || reading}
          onClick={() => fileRef.current?.click()}
        >
          {reading ? <Loader2 className="animate-spin" /> : <Upload />}上传文件
        </Button>
        <span className="text-xs text-muted-foreground">
          TXT / Markdown，最多 2000 万字
        </span>
        {hasLegacySource && (
          <Button
            variant="ghost"
            disabled={disabled || reading}
            onClick={onLegacySource}
          >
            拆分已保存原文
          </Button>
        )}
      </div>
      {file && (
        <div className="flex min-w-0 flex-col gap-2">
          <div className="flex min-w-0 items-center gap-2 text-sm">
            <FileText className="size-4 shrink-0" />
            <span className="truncate">{file.name}</span>
            <Button
              variant="ghost"
              disabled={disabled || reading}
              onClick={() => {
                setFile(null);
                onSource("");
                onError("");
              }}
            >
              移除
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <label className="shrink-0 text-sm" htmlFor="source-encoding">
              文件编码
            </label>
            <Select
              items={encodings}
              value={encoding}
              disabled={disabled || reading}
              onValueChange={(value) => {
                if (value) {
                  setEncoding(value);
                  void read(file, value);
                }
              }}
            >
              <SelectTrigger id="source-encoding" className="w-48">
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
        </div>
      )}
      <div className="flex min-w-0 flex-col gap-2">
        <label htmlFor="script-source" className="text-sm font-medium">
          故事原文
        </label>
        {file ? (
          <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">
            {reading
              ? "正在读取文件…"
              : source
                ? `已读取 ${source.length.toLocaleString()} 字，下一步检查分集。`
                : "文件读取失败"}
          </div>
        ) : (
          <Textarea
            id="script-source"
            value={source}
            onChange={(event) => {
              onSource(event.target.value);
              onError("");
            }}
            rows={10}
            maxLength={MAX_SOURCE_LENGTH}
            disabled={disabled || reading}
            className="field-sizing-fixed min-h-48 w-full resize-y"
            placeholder="粘贴小说、故事或剧本原文"
          />
        )}
        {!file && (
          <p className="text-xs text-muted-foreground">
            {source.length.toLocaleString()} /{" "}
            {MAX_SOURCE_LENGTH.toLocaleString()} 字
          </p>
        )}
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <label htmlFor="source-rule" className="text-sm font-medium">
            分集方式
          </label>
          <Select
            items={rules}
            value={rule}
            disabled={disabled || reading}
            onValueChange={(value) => {
              if (value) {
                onRule(value as ScriptSplitRule);
              }
            }}
          >
            <SelectTrigger id="source-rule" className="w-full">
              <SelectValue placeholder="选择分集方式" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {rules.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <label htmlFor="source-length" className="text-sm font-medium">
            每集最多字数
          </label>
          <Input
            id="source-length"
            type="number"
            min={1000}
            max={MAX_EPISODE_LENGTH}
            step={1000}
            value={targetLength || ""}
            disabled={disabled || reading}
            onChange={(event) => onTargetLength(Number(event.target.value))}
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        识别“第十二章”“第 3 集”“Chapter IV”等标题。长章节会继续按段落拆分。
      </p>
    </div>
  );
}
