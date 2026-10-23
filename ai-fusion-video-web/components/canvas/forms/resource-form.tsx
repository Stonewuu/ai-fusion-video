"use client";

import { useMemo } from "react";
import { AssetPropertiesEditor } from "@/components/dashboard/asset-detail/asset-properties-editor";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";
import { resourceFieldSection } from "@/lib/canvas/resource-field-presentation";
import type { ResourceType } from "@/lib/api/resource-events";
import type { CanvasLibraryEntry } from "@/lib/canvas/resource-library";
import { CanvasResourceFieldControl } from "./resource-field-control";

export function CanvasResourceForm({
  type,
  data,
  onChange,
  library,
  disabled = false,
  assetType,
}: {
  type: ResourceType;
  data: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
  library: CanvasLibraryEntry[];
  disabled?: boolean;
  assetType?: string;
}) {
  const sections = useMemo(() => {
    const result = new Map<string, (typeof RESOURCE_FIELDS)[ResourceType]>();
    RESOURCE_FIELDS[type].forEach((field) => {
      const section = resourceFieldSection(type, field);
      result.set(section, [...(result.get(section) ?? []), field]);
    });
    return [...result.entries()];
  }, [type]);
  const properties =
    data.properties &&
    typeof data.properties === "object" &&
    !Array.isArray(data.properties)
      ? (data.properties as Record<string, unknown>)
      : {};
  return (
    <div className="flex min-w-0 flex-col gap-5">
      {sections.map(([section, fields]) => (
        <details key={section} open={section === "基本内容"} className="group min-w-0 border-b border-border/20 pb-4">
          <summary className="cursor-pointer rounded-lg text-sm font-semibold outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
            {section}
            <span className="ml-2 text-xs font-normal text-muted-foreground">{fields.filter((field) => data[field.key] != null && data[field.key] !== "").length} / {fields.length}</span>
          </summary>
          <div className="flex min-w-0 flex-col gap-4 pt-4">
            {fields.map((field) => (
              <CanvasResourceFieldControl
                key={field.key}
                field={field}
                value={data[field.key]}
                disabled={disabled}
                library={library}
                onChange={(value) => onChange(field.key, value)}
              />
            ))}
          </div>
        </details>
      ))}
      {((type === "asset" && typeof data.type === "string") ||
        (type === "assetItem" && assetType)) && (
        <AssetPropertiesEditor
          assetType={String(type === "asset" ? data.type : assetType)}
          value={properties}
          disabled={disabled}
          onChange={(key, value) =>
            onChange("properties", { ...properties, [key]: value })
          }
        />
      )}
    </div>
  );
}
