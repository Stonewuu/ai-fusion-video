"use client";

import { SlidersHorizontal } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ScriptSplitRule } from "@/lib/api/script";
import { MAX_EPISODE_LENGTH } from "@/lib/scripts/source-import-limits";

const rules = [
  { value: "AUTO", label: "自动识别" },
  { value: "CHAPTER", label: "按章节标题" },
  { value: "EPISODE", label: "按分集标题" },
  { value: "LENGTH", label: "按字数分集" },
];

const ruleHints: Record<ScriptSplitRule, string> = {
  AUTO: "识别章节和分集标题，如“第十二章”“第 3 集”“Chapter IV”。",
  CHAPTER: "按“第十二章”“Chapter IV”等章节标题拆分。",
  EPISODE: "按“第 3 集”“Episode 3”等分集标题拆分。",
  LENGTH: "按字数拆分，并尽量保留完整段落。",
};

interface Props {
  rule: ScriptSplitRule;
  targetLength: number;
  disabled: boolean;
  onRule: (rule: ScriptSplitRule) => void;
  onTargetLength: (length: number) => void;
}

export function SourceInputSettings({
  rule,
  targetLength,
  disabled,
  onRule,
  onTargetLength,
}: Props) {
  const invalidLength =
    !Number.isInteger(targetLength) ||
    targetLength < 1000 ||
    targetLength > MAX_EPISODE_LENGTH;

  return (
    <aside className="flex min-w-0 flex-col gap-5 self-start rounded-xl bg-muted/35 p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <SlidersHorizontal className="size-4 text-muted-foreground" />
        拆分设置
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <label htmlFor="source-rule" className="text-sm font-medium">
          分集方式
        </label>
        <Select
          items={rules}
          value={rule}
          disabled={disabled}
          onValueChange={(value) => {
            if (value) {
              onRule(value as ScriptSplitRule);
            }
          }}
        >
          <SelectTrigger
            id="source-rule"
            className="w-full"
            aria-describedby="source-rule-hint"
          >
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
        <p
          id="source-rule-hint"
          className="text-xs leading-5 text-muted-foreground"
        >
          {ruleHints[rule]}
        </p>
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
          disabled={disabled}
          aria-invalid={invalidLength}
          aria-describedby={
            invalidLength ? "source-length-error" : "source-length-hint"
          }
          onChange={(event) => onTargetLength(Number(event.target.value))}
        />
        {invalidLength ? (
          <p
            id="source-length-error"
            role="alert"
            className="text-xs leading-5 text-destructive"
          >
            请输入 1,000 至 100,000 之间的字数。
          </p>
        ) : (
          <p
            id="source-length-hint"
            className="text-xs leading-5 text-muted-foreground"
          >
            超过字数的章节会继续按段落拆分。
          </p>
        )}
      </div>
      <p className="border-t border-border/20 pt-4 text-xs leading-5 text-muted-foreground">
        下一步可修改标题、调整分界，或删除不需要的内容。
      </p>
    </aside>
  );
}
