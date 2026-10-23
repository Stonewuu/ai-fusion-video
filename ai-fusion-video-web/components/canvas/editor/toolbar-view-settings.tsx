"use client";

import { useRef, useState } from "react";
import {
  Map,
  Scan,
  Search,
  SlidersHorizontal,
  Wand2,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { CanvasRelationScope } from "@/lib/canvas/relations";
import { RelationVisibilityControls } from "./relation-visibility-controls";

export function CanvasToolbarAction({
  icon: Icon,
  label,
  title,
  onClick,
  disabled,
  primary = false,
}: {
  icon: LucideIcon;
  label: string;
  title?: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <>
      <span className="@3xl:hidden">
        <Button
          size="icon"
          variant={primary ? "default" : "ghost"}
          aria-label={label}
          data-canvas-add={primary || undefined}
          title={title ?? label}
          onClick={onClick}
          disabled={disabled}
        >
          <Icon />
        </Button>
      </span>
      <span className="hidden @3xl:inline-flex">
        <Button
          size="default"
          variant={primary ? "default" : "ghost"}
          data-canvas-add={primary || undefined}
          title={title ?? label}
          onClick={onClick}
          disabled={disabled}
        >
          <Icon />
          {label}
        </Button>
      </span>
    </>
  );
}

export function CanvasViewActions({
  onSearch,
  onFit,
  onArrange,
  editable,
}: {
  onSearch: () => void;
  onFit: () => void;
  onArrange: () => void;
  editable: boolean;
}) {
  return (
    <>
      <span className="hidden @md:inline-flex">
        <CanvasToolbarAction
          icon={Search}
          label="查找"
          title="查找并定位节点（/）"
          onClick={onSearch}
        />
      </span>
      <CanvasToolbarAction
        icon={Scan}
        label="查看全部"
        title="查看全部节点（F）"
        onClick={onFit}
      />
      <span className="hidden @md:inline-flex">
        <CanvasToolbarAction
          icon={Wand2}
          label="自动整理"
          title="按连线从左到右整理，优先保持同一链路水平"
          onClick={onArrange}
          disabled={!editable}
        />
      </span>
    </>
  );
}

function MinimapToggle({ checked, onToggle, fullWidth = false }: {
  checked: boolean;
  onToggle: () => void;
  fullWidth?: boolean;
}) {
  return (
    <Button
      variant="secondary"
      size="default"
      aria-pressed={checked}
      className={fullWidth ? "w-full justify-start" : undefined}
      onClick={onToggle}
    >
      <Map aria-hidden />
      {checked ? "隐藏小地图" : "显示小地图"}
    </Button>
  );
}

export function CanvasViewSettings({
  relations,
  onRelations,
  minimap,
  onMinimap,
  onSearch,
  onArrange,
  editable,
}: {
  relations: CanvasRelationScope;
  onRelations: (scope: CanvasRelationScope) => void;
  minimap: boolean;
  onMinimap: () => void;
  onSearch: () => void;
  onArrange: () => void;
  editable: boolean;
}) {
  const [open, setOpen] = useState(false);
  const handoffFocus = useRef(false);
  return (
    <div className="pointer-events-auto shrink-0 rounded-2xl border border-border/40 bg-popover/90 p-1 shadow-sm backdrop-blur-xl">
      <div className="hidden items-center gap-2 @5xl:flex">
        <MinimapToggle checked={minimap} onToggle={onMinimap} />
        <RelationVisibilityControls
          value={relations}
          onChange={onRelations}
        />
      </div>
      <div className="@5xl:hidden">
        <Popover
          open={open}
          onOpenChange={(next) => {
            if (next) handoffFocus.current = false;
            setOpen(next);
          }}
        >
          <PopoverTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                aria-label="显示与工具"
                title="显示与工具"
              />
            }
          >
            <SlidersHorizontal />
          </PopoverTrigger>
          <PopoverContent
            align="end"
            finalFocus={() => !handoffFocus.current}
            className="w-80 max-w-[calc(100vw-2rem)] border border-border/40"
          >
            <PopoverTitle>显示与工具</PopoverTitle>
            <RelationVisibilityControls value={relations} onChange={onRelations} expanded />
            <MinimapToggle checked={minimap} onToggle={onMinimap} fullWidth />
            <div className="grid grid-cols-2 gap-2 border-t border-border/20 pt-4">
              <Button
                variant="outline"
                onClick={() => {
                  handoffFocus.current = true;
                  setOpen(false);
                  onSearch();
                }}
              >
                <Search />
                查找
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setOpen(false);
                  onArrange();
                }}
                disabled={!editable}
              >
                <Wand2 />
                自动整理
              </Button>
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </div>
  );
}
