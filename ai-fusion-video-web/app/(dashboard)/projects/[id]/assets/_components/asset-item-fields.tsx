"use client";

import { useId, type Ref } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { AssetItemMessage } from "./asset-item-feedback";

interface Props {
  name: string;
  appearance: string;
  disabled: boolean;
  invalidName: boolean;
  nameRef: Ref<HTMLInputElement>;
  textareaRef: Ref<HTMLTextAreaElement>;
  hints: readonly string[];
  onName: (value: string) => void;
  onNameBlur: () => void;
  onAppearance: (value: string) => void;
  onHint: (hint: string) => void;
}

export function AssetItemFields({
  name,
  appearance,
  disabled,
  invalidName,
  nameRef,
  textareaRef,
  hints,
  onName,
  onNameBlur,
  onAppearance,
  onHint,
}: Props) {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-4 border-b border-border/20 p-5 md:border-b-0 md:border-l">
      <div className="flex min-w-0 flex-col gap-2">
        <label htmlFor={`${id}-name`} className="text-sm font-medium">
          名称
        </label>
        <div>
          <Input
            ref={nameRef}
            id={`${id}-name`}
            disabled={disabled}
            value={name}
            onChange={(event) => onName(event.target.value)}
            onBlur={onNameBlur}
            aria-invalid={invalidName}
            aria-describedby={invalidName ? `${id}-name-error` : undefined}
            placeholder="子资产名称"
          />
          <AssetItemMessage
            id={`${id}-name-error`}
            message={invalidName ? "请输入子资产名称" : ""}
          />
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <label htmlFor={`${id}-appearance`} className="text-sm font-medium">
            外观描写
          </label>
          <span className="text-xs tabular-nums text-muted-foreground">
            {appearance.length} 字
          </span>
        </div>
        <Textarea
          ref={textareaRef}
          id={`${id}-appearance`}
          disabled={disabled}
          value={appearance}
          onChange={(event) => onAppearance(event.target.value)}
          rows={8}
          placeholder="描述材质、颜色、形态和细节"
          className="field-sizing-fixed min-h-40 resize-y text-sm leading-6"
        />
      </div>
      <div className="flex flex-col gap-2">
        <p className="text-xs text-muted-foreground">描写要点</p>
        <div className="flex flex-wrap gap-2">
          {hints.map((hint) => (
            <button
              key={hint}
              type="button"
              disabled={disabled}
              onClick={() => onHint(hint)}
              className="h-6 rounded-md border border-border/30 bg-background/70 px-2 text-xs text-muted-foreground transition-colors duration-150 hover:border-primary/30 hover:bg-primary/5 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 motion-reduce:transition-none"
            >
              + {hint}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
