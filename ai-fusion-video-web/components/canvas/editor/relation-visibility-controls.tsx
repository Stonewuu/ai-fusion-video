"use client";

import { useId } from "react";
import { Square, SquareCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { toggleCanvasRelation, type CanvasRelationScope } from "@/lib/canvas/relations";

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

export function RelationVisibilityControls({ value, onChange, expanded = false }: {
  value: CanvasRelationScope;
  onChange: (scope: CanvasRelationScope) => void;
  expanded?: boolean;
}) {
  const descriptionPrefix = useId();
  return (
    <div role="group" aria-label="辅助连线" className={expanded ? "flex flex-col gap-3" : "flex items-center gap-2"}>
      <span className={expanded ? "text-xs text-muted-foreground" : "sr-only"}>辅助连线</span>
      <TooltipProvider delay={300}>
        {options.map((option) => {
          const checked = value === "all" || value === option.value;
          const Icon = checked ? SquareCheck : Square;
          const action = checked ? "隐藏" : "显示";
          const descriptionId = `${descriptionPrefix}-${option.value}`;
          const button = <Button
            variant={checked ? "secondary" : "ghost"}
            size="default"
            role="checkbox"
            aria-label={`${option.label}连线`}
            aria-checked={checked}
            aria-describedby={descriptionId}
            className={expanded ? "w-full justify-start" : undefined}
            onClick={() => onChange(toggleCanvasRelation(value, option.value))}
          >
            <Icon aria-hidden />
            {option.label}
          </Button>;
          return expanded ? (
            <div key={option.value} className="flex flex-col gap-1">
              {button}
              <p id={descriptionId} className="text-xs leading-relaxed text-muted-foreground">{option.description}</p>
            </div>
          ) : (
            <Tooltip key={option.value}>
              <TooltipTrigger render={button} />
              <TooltipContent id={descriptionId} role="tooltip" side="bottom" className="flex-col items-start gap-1">
                <span className="font-medium">{action}{option.label}连线</span>
                <p>{option.description}</p>
                <p>可同时勾选，只显示画布内的关联。创作连线保持不变。</p>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </TooltipProvider>
      {expanded && <p className="text-xs leading-relaxed text-muted-foreground">可同时勾选，只显示画布内的关联。创作连线保持不变。</p>}
    </div>
  );
}
