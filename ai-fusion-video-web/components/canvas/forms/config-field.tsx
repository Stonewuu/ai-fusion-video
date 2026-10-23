"use client";

import { useId } from "react";
import { Link2, Monitor, Smartphone, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type {
  CanvasConfig,
  CanvasConfigField,
  CanvasModelOption,
} from "@/lib/canvas/types";
import { canvasFieldLabels } from "@/lib/canvas/labels";
import {
  configFieldChoices,
  configFieldError,
  configNumberBounds,
  fieldHints,
} from "@/lib/canvas/field-presentation";
import {
  CanvasBooleanField,
  CanvasChoiceField,
  CanvasFieldReset,
  CanvasNumberField,
} from "./field-controls";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { videoDurationOptions, videoRulesOf, type VideoInputState } from "@/lib/video-model-rules";
import { GenerationDurationSlider } from "@/components/dashboard/generation/generation-duration-slider";

export function CanvasConfigFieldEditor({
  kind,
  name,
  field,
  config,
  model,
  connected,
  videoInputs,
  disabled,
  onChange,
}: {
  kind: string;
  name: string;
  field: CanvasConfigField;
  config: CanvasConfig;
  model?: CanvasModelOption;
  connected: boolean;
  videoInputs?: VideoInputState;
  disabled: boolean;
  onChange: (name: string, value: CanvasConfig[string] | undefined) => void;
}) {
  const id = useId();
  const value = config[name];
  const promptRules =
    name === "prompt" ? videoRulesOf(model?.parameters) : undefined;
  const textLimit = promptRules?.promptMaxLength ?? field.maxLength;
  const label = canvasFieldLabels[name] ?? name;
  const choices = configFieldChoices(kind, name, field, model, videoInputs);
  const bounds = configNumberBounds(name, field, model, config);
  const error = configFieldError(name, field, value, choices, model, config);
  const volume = ["originalVolume", "musicVolume", "sfxVolume"].includes(name);
  const durationOptions = videoDurationOptions(model?.parameters, String(config.resolution ?? model?.parameters.defaultResolution ?? ""));
  const number = typeof value === "number" ? value : undefined;
  const update = (next: CanvasConfig[string] | undefined) =>
    onChange(name, next);
  const customChoice =
    name === "separator" || (kind === "videoCompose" && name === "resolution");
  return (
    <div className="flex min-w-0 flex-col gap-2" data-canvas-field={name}>
      <div className="flex min-h-6 items-center justify-between gap-2">
        <label id={id + "-label"} htmlFor={id} className="text-sm font-medium">
          {label}
          {field.required ? " *" : ""}
        </label>
        {!field.required &&
          value !== undefined &&
          !connected &&
          field.type !== "boolean" && (
            <CanvasFieldReset
              label={label}
              disabled={disabled}
              onReset={() => update(undefined)}
            />
          )}
      </div>
      {connected ? (
        <p className="flex items-center gap-2 text-xs text-primary">
          <Link2 className="size-4 shrink-0" />
          已连接上游内容
        </p>
      ) : name === "duration" &&
        (videoRulesOf(model?.parameters)?.automaticDuration || durationOptions?.length) ? (
        <fieldset disabled={disabled} className="min-w-0">
          <GenerationDurationSlider
            value={number ?? Number(model?.parameters.defaultDuration)}
            min={bounds.min!}
            max={bounds.max!}
            automatic={videoRulesOf(model?.parameters)?.automaticDuration}
            options={durationOptions}
            onChange={update}
          />
        </fieldset>
      ) : name === "ratio" && choices.length ? (
        <div
          className="grid grid-cols-3 gap-2"
          role="group"
          aria-labelledby={id + "-label"}
        >
          {choices.map((choice) => {
            const [width, height] = choice.value.split(":").map(Number);
            const Icon =
              width === height ? Square : width > height ? Monitor : Smartphone;
            return (
              <Button
                key={choice.value}
                variant={value === choice.value ? "secondary" : "outline"}
                size="default"
                aria-pressed={value === choice.value}
                disabled={disabled}
                onClick={() => update(choice.value)}
              >
                <Icon />
                {choice.label}
              </Button>
            );
          })}
          {value !== undefined &&
            !choices.some((item) => item.value === value) && (
              <p className="col-span-3 text-xs text-destructive">
                当前：{String(value)}
              </p>
            )}
        </div>
      ) : choices.length ? (
        <CanvasChoiceField
          label={label}
          value={value === undefined ? null : String(value)}
          items={choices}
          custom={customChoice}
          disabled={disabled}
          invalid={!!error}
          placeholder={"选择" + label}
          onChange={update}
        />
      ) : field.type === "boolean" ? (
        <CanvasBooleanField
          label={label}
          value={typeof value === "boolean" ? value : undefined}
          disabled={disabled}
          unsupported={
            name === "cameraFixed" &&
            model?.parameters.supportCameraFixed === false
          }
          onChange={update}
        />
      ) : name === "seed" ? (
        <div className="flex flex-col gap-3">
          <CanvasSegmentedControl
            label="种子模式"
            value={value === undefined || value === -1 ? "random" : "fixed"}
            className="w-full"
            options={[
              {
                value: "random",
                label: "每次随机",
                disabled,
              },
              { value: "fixed", label: "固定种子", disabled },
            ]}
            onChange={(mode) => update(mode === "random" ? undefined : 0)}
          />
          {value !== undefined && value !== -1 && (
            <CanvasNumberField
              id={id}
              label={label}
              value={number}
              min={0}
              max={field.maximum ?? undefined}
              disabled={disabled}
              onChange={update}
              invalid={!!error}
            />
          )}
        </div>
      ) : field.type === "number" || field.type === "integer" ? (
        <CanvasNumberField
          id={id}
          label={label}
          value={
            volume && number !== undefined ? Math.round(number * 100) : number
          }
          min={volume ? 0 : bounds.min}
          max={volume ? 100 : bounds.max}
          step={volume ? 1 : field.type === "integer" ? 1 : 0.05}
          unit={
            volume
              ? "%"
              : name === "duration"
                ? "秒"
                : name === "count"
                  ? "个"
                  : ["width", "height", "x", "y"].includes(name)
                    ? "px"
                    : undefined
          }
          disabled={disabled}
          invalid={!!error}
          range={
            volume ||
            (name === "duration" &&
              bounds.max !== undefined &&
              bounds.max <= 120)
          }
          placeholder={
            name === "duration" && model?.parameters.defaultDuration
              ? "默认 " + model.parameters.defaultDuration + " 秒"
              : "默认"
          }
          presets={
            name === "count"
              ? [1, 2, 4]
              : name === "frameRate" || name === "fps"
                ? [24, 25, 30, 60]
                : volume
                  ? [0, 25, 50, 100]
                  : []
          }
          onChange={(next) =>
            update(volume && next !== undefined ? next / 100 : next)
          }
        />
      ) : (field.maxLength ?? 0) > 256 ? (
        <div className="flex flex-col gap-2">
          <Textarea
            id={id}
            value={String(value ?? "")}
            rows={name === "content" ? 8 : 4}
            disabled={disabled}
            placeholder={
              name === "content"
                ? kind === "note"
                  ? "记录想法或给团队留一句话…"
                  : "在这里写下故事、文字或创作灵感…"
                : fieldHints[name]
            }
            maxLength={promptRules ? undefined : (field.maxLength ?? undefined)}
            onChange={(event) => update(event.target.value)}
          />
          <span className="text-right text-xs text-muted-foreground">
            {[...String(value ?? "")].length.toLocaleString()} /{" "}
            {textLimit?.toLocaleString()}
          </span>
        </div>
      ) : (
        <Input
          id={id}
          value={String(value ?? "")}
          disabled={disabled}
          placeholder={"填写" + label}
          maxLength={field.maxLength ?? undefined}
          aria-invalid={!!error || undefined}
          onChange={(event) => update(event.target.value)}
        />
      )}
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
      {!connected &&
        fieldHints[name] &&
        !["prompt", "instruction", "negativePrompt"].includes(name) && (
          <p className="text-xs leading-relaxed text-muted-foreground">
            {fieldHints[name]}
          </p>
        )}
    </div>
  );
}
