"use client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { assetApi, type FieldDef } from "@/lib/api/asset";
import { Loader2 } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { LONG_TEXT_KEYS, propertyText } from "./shared";
export function AssetPropertiesEditor({
  assetType,
  value,
  disabled,
  onChange,
}: {
  assetType: string;
  value: Record<string, unknown>;
  disabled?: boolean;
  onChange: (key: string, value: string | null) => void;
}) {
  const id = useId();
  const [attempt, setAttempt] = useState(0);
  const requestKey = assetType + ":" + attempt;
  const [result, setResult] = useState<{
    key: string;
    fields: FieldDef[];
    error: string;
  } | null>(null);
  const loading = result?.key !== requestKey;
  const fields = loading ? [] : result.fields;
  const error = loading ? "" : result.error;
  useEffect(() => {
    let active = true;
    assetApi
      .getMetadata(assetType)
      .then((data) => {
        if (active)
          setResult({ key: requestKey, fields: data.fields, error: "" });
      })
      .catch((failure) => {
        if (active)
          setResult({
            key: requestKey,
            fields: [],
            error: getApiErrorMessage(failure),
          });
      });
    return () => {
      active = false;
    };
  }, [assetType, requestKey]);
  return (
    <section className="space-y-3" aria-busy={loading}>
      <h4 className="flex items-center gap-2 text-sm font-medium">
        属性
        {loading && <Loader2 className="size-4 animate-spin" />}
      </h4>
      {error ? (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>{error}</p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAttempt((current) => current + 1)}
          >
            重试
          </Button>
        </div>
      ) : loading ? (
        <p className="text-sm text-muted-foreground">正在加载属性</p>
      ) : fields.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          此类型使用扩展属性；已有内容会保留。
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-3">
          {fields.map((field) => (
            <div key={field.key} className="flex min-w-0 flex-col gap-2">
              <label
                htmlFor={`${id}-${field.key}`}
                className="text-sm text-muted-foreground"
              >
                {field.label}
              </label>
              {field.type === "select" && field.options ? (
                <Select
                  disabled={disabled}
                  items={field.options}
                  value={propertyText(value[field.key])}
                  onValueChange={(next) => onChange(field.key, next || null)}
                >
                  <SelectTrigger id={`${id}-${field.key}`} className="text-sm">
                    <SelectValue placeholder="选择" />
                  </SelectTrigger>
                  <SelectContent className="text-sm">
                    <SelectGroup>
                      {field.options.map((option) => (
                        <SelectItem
                          key={option.value}
                          value={option.value}
                          className="text-sm"
                        >
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              ) : LONG_TEXT_KEYS.has(field.key) ? (
                <Textarea
                  id={`${id}-${field.key}`}
                  disabled={disabled}
                  value={propertyText(value[field.key])}
                  onChange={(event) => onChange(field.key, event.target.value)}
                  rows={3}
                />
              ) : (
                <Input
                  id={`${id}-${field.key}`}
                  disabled={disabled}
                  value={propertyText(value[field.key])}
                  onChange={(event) => onChange(field.key, event.target.value)}
                />
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
