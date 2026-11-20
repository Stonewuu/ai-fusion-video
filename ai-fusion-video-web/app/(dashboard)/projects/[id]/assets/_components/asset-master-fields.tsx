"use client";

import { useId, type Ref } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
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
import { ASSET_TYPES } from "./asset-types";

const typeOptions = ASSET_TYPES.map(({ value, label }) => ({ value, label }));

interface Props {
  name: string;
  type: string;
  description: string;
  disabled: boolean;
  showNameError: boolean;
  nameRef?: Ref<HTMLInputElement>;
  autoFocus?: boolean;
  onName: (value: string) => void;
  onNameBlur: () => void;
  onType: (value: string) => void;
  onDescription: (value: string) => void;
}

export function AssetMasterFields({
  name,
  type,
  description,
  disabled,
  showNameError,
  nameRef,
  autoFocus = false,
  onName,
  onNameBlur,
  onType,
  onDescription,
}: Props) {
  const id = useId();

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="grid min-w-0 gap-4 sm:grid-cols-[minmax(0,1fr)_10rem]">
        <div className="flex min-w-0 flex-col gap-2">
          <label htmlFor={`${id}-name`} className="text-sm font-medium">
            资产名称
          </label>
          <div className="min-w-0">
            <Input
              ref={nameRef}
              id={`${id}-name`}
              value={name}
              disabled={disabled}
              autoFocus={autoFocus}
              aria-required="true"
              aria-invalid={showNameError}
              aria-describedby={showNameError ? `${id}-name-error` : undefined}
              onChange={(event) => onName(event.target.value)}
              onBlur={onNameBlur}
              placeholder="填写资产名称"
              className="bg-background/80"
            />
            <AssetMasterFeedback
              id={`${id}-name-error`}
              message={showNameError ? "请输入资产名称" : ""}
              compact
            />
          </div>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <label htmlFor={`${id}-type`} className="text-sm font-medium">
            资产类型
          </label>
          <Select
            items={typeOptions}
            value={type}
            disabled={disabled}
            onValueChange={(value) => {
              if (value) {
                onType(value);
              }
            }}
          >
            <SelectTrigger
              id={`${id}-type`}
              className="w-full bg-background/80"
            >
              <SelectValue placeholder="选择资产类型" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {typeOptions.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <label htmlFor={`${id}-description`} className="text-sm font-medium">
          资产设定
        </label>
        <Textarea
          id={`${id}-description`}
          value={description}
          disabled={disabled}
          onChange={(event) => onDescription(event.target.value)}
          rows={4}
          placeholder="填写资产的背景与设定"
          className="field-sizing-fixed resize-none bg-background/80 text-sm leading-6"
        />
      </div>
    </div>
  );
}

export function AssetMasterFeedback({
  message,
  id,
  compact = false,
}: {
  message: string;
  id?: string;
  compact?: boolean;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <AnimatePresence initial={false}>
      {message && (
        <motion.div
          key="feedback"
          initial={reduceMotion ? false : { height: 0, opacity: 0 }}
          animate={{
            height: "auto",
            opacity: 1,
            transition: { duration: reduceMotion ? 0 : 0.12, ease: "easeOut" },
          }}
          exit={{
            height: 0,
            opacity: 0,
            transition: { duration: reduceMotion ? 0 : 0.08, ease: "easeIn" },
          }}
          className="overflow-hidden"
        >
          <p
            id={id}
            role="alert"
            className={
              compact
                ? "pt-2 text-xs text-destructive"
                : "pt-4 text-sm text-destructive"
            }
          >
            {message}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
