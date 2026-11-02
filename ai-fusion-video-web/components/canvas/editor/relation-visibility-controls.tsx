"use client";

import { useId } from "react";
import { Square, SquareCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  toggleCanvasRelation,
  type CanvasRelationAvailability,
  type CanvasRelationScope,
} from "@/lib/canvas/relations";

const options = [
  {
    value: "structure", label: "包含内容",
    description: "查看分集里的场次、场次里的镜头，以及资产的不同版本。",
  },
  {
    value: "assets", label: "素材与来源",
    description: "查看镜头用到的角色、场景、道具，以及分镜来自哪集剧本。",
  },
] as const;

export function RelationVisibilityControls({ value, onChange, available }: {
  value: CanvasRelationScope;
  onChange: (scope: CanvasRelationScope) => void;
  available: CanvasRelationAvailability;
}) {
  const descriptionPrefix = useId();
  const visibleOptions = options.filter((option) => available[option.value]);
  if (!visibleOptions.length) {
    return null;
  }
  return (
    <div role="group" aria-label="项目关系线" className="flex flex-col gap-3">
      <p className="text-xs font-medium text-muted-foreground">项目关系线</p>
      {visibleOptions.map((option) => {
        const checked = value === "all" || value === option.value;
        const Icon = checked ? SquareCheck : Square;
        const descriptionId = `${descriptionPrefix}-${option.value}`;
        return (
          <div key={option.value} className="flex flex-col gap-1">
            <Button
              variant={checked ? "secondary" : "ghost"}
              size="default"
              role="checkbox"
              aria-label={`${option.label}连线`}
              aria-checked={checked}
              aria-describedby={descriptionId}
              className="w-full justify-start"
              onClick={() => onChange(toggleCanvasRelation(value, option.value))}
            >
              <Icon aria-hidden />
              {option.label}
            </Button>
            <p id={descriptionId} className="text-xs leading-relaxed text-muted-foreground">
              {option.description}
            </p>
          </div>
        );
      })}
    </div>
  );
}
