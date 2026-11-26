"use client";

import { useRef, useState } from "react";
import { FileText, History, TextCursorInput, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ScriptSplitRule } from "@/lib/api/script";
import { readSourceFile } from "@/lib/scripts/source-import";
import { MAX_SOURCE_LENGTH } from "@/lib/scripts/source-import-limits";
import { SourceInputFile } from "./source-input-file";
import { SourceInputSettings } from "./source-input-settings";

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
  onFile,
  onEncoding,
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
  const [inputMode, setInputMode] = useState(
    file || !source ? "file" : "paste",
  );

  const read = async (nextFile: File, nextEncoding: string) => {
    onFile(nextFile);
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
    <div className="grid min-w-0 gap-6 md:grid-cols-[minmax(0,1fr)_15rem]">
      <div className="flex min-w-0 flex-col gap-4 p-1">
        <input
          ref={fileRef}
          type="file"
          accept=".txt,.md,.markdown"
          className="hidden"
          aria-label="选择原文文件"
          disabled={disabled || reading}
          onChange={(event) => {
            const nextFile = event.target.files?.[0];
            if (nextFile) {
              void read(nextFile, encoding);
            }
            event.target.value = "";
          }}
        />
        <Tabs
          value={file ? "file" : inputMode}
          onValueChange={(value) => setInputMode(String(value))}
          className="min-w-0 gap-4"
        >
          <TabsList aria-label="原文来源" className="min-h-11 p-1">
            <TabsTrigger
              value="file"
              disabled={disabled || reading}
              className="h-9 px-3 transition-colors motion-reduce:transition-none"
            >
              <Upload />
              上传文件
            </TabsTrigger>
            <TabsTrigger
              value="paste"
              disabled={disabled || reading || !!file}
              title={file ? "移除文件后可粘贴原文" : undefined}
              className="h-9 px-3 transition-colors motion-reduce:transition-none"
            >
              <TextCursorInput />
              粘贴原文
            </TabsTrigger>
          </TabsList>
          <TabsContent value="file" className="min-w-0">
            <SourceInputFile
              file={file}
              source={source}
              encoding={encoding}
              reading={reading}
              disabled={disabled}
              onChoose={() => fileRef.current?.click()}
              onRemove={() => {
                onFile(null);
                onSource("");
                onError("");
              }}
              onEncoding={(value) => {
                onEncoding(value);
                if (file) {
                  void read(file, value);
                }
              }}
            />
          </TabsContent>
          <TabsContent value="paste" className="min-w-0">
            <div className="flex min-w-0 flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <label htmlFor="script-source" className="text-sm font-medium">
                  故事原文
                </label>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {source.length.toLocaleString()} 字
                </span>
              </div>
              <Textarea
                id="script-source"
                value={source}
                onChange={(event) => {
                  onSource(event.target.value);
                  onError("");
                }}
                rows={13}
                maxLength={MAX_SOURCE_LENGTH}
                disabled={disabled || reading}
                className="field-sizing-fixed min-h-64 w-full resize-y text-sm leading-7"
                placeholder="粘贴小说、故事或剧本原文，保留章节标题和段落。"
                aria-describedby="source-paste-hint"
              />
              <p
                id="source-paste-hint"
                className="text-xs text-muted-foreground"
              >
                最多 2000 万字，长篇内容建议上传文件。
              </p>
            </div>
          </TabsContent>
        </Tabs>
        {hasLegacySource && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/20 pt-3">
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <FileText className="size-4" />
              剧本中已有原文
            </span>
            <Button
              variant="ghost"
              disabled={disabled || reading}
              onClick={() => {
                setInputMode("paste");
                onLegacySource();
              }}
            >
              <History />
              拆分已保存原文
            </Button>
          </div>
        )}
      </div>
      <SourceInputSettings
        rule={rule}
        targetLength={targetLength}
        disabled={disabled || reading}
        onRule={onRule}
        onTargetLength={onTargetLength}
      />
    </div>
  );
}
