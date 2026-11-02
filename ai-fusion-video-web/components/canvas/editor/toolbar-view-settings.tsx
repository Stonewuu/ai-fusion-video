"use client";

import { useRef, useState } from "react";
import {
  Map,
  Scan,
  Search,
  SlidersHorizontal,
  Wand2,
  X,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { CanvasRelationAvailability, CanvasRelationScope } from "@/lib/canvas/relations";
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
  arranging,
}: {
  onSearch: () => void;
  onFit: () => void;
  onArrange: () => void;
  editable: boolean;
  arranging: boolean;
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
          icon={arranging ? X : Wand2}
          label={arranging ? "取消整理" : "自动整理"}
          title={arranging ? "正在整理，点击取消" : "按执行链路和项目关系整理选区，未选择时整理全部"}
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
  relationAvailability,
  relationLoading,
  relationError,
  onReloadRelations,
  minimap,
  onMinimap,
  onSearch,
  onArrange,
  editable,
  arranging,
}: {
  relations: CanvasRelationScope;
  onRelations: (scope: CanvasRelationScope) => void;
  relationAvailability: CanvasRelationAvailability;
  relationLoading: boolean;
  relationError: string | null;
  onReloadRelations: () => void;
  minimap: boolean;
  onMinimap: () => void;
  onSearch: () => void;
  onArrange: () => void;
  editable: boolean;
  arranging: boolean;
}) {
  const [open, setOpen] = useState(false);
  const handoffFocus = useRef(false);
  return (
    <div className="pointer-events-auto flex shrink-0 items-center gap-2 rounded-2xl border border-border/40 bg-popover/90 p-1 shadow-sm backdrop-blur-xl">
      <div className="hidden @5xl:block">
        <MinimapToggle checked={minimap} onToggle={onMinimap} />
      </div>
      <Popover
        open={open}
        onOpenChange={(next) => {
          if (next) {
            handoffFocus.current = false;
          }
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
          className="max-h-[calc(100vh-2rem)] w-80 max-w-[calc(100vw-2rem)] overflow-y-auto border border-border/40"
        >
          <PopoverTitle>显示与工具</PopoverTitle>
          <MinimapToggle checked={minimap} onToggle={onMinimap} fullWidth />
          {relationLoading && (
            <p role="status" className="text-xs text-muted-foreground">正在读取项目关系…</p>
          )}
          {relationError && (
            <div className="flex flex-col gap-2">
              <p role="alert" className="text-xs text-destructive">{relationError}</p>
              <Button variant="outline" onClick={onReloadRelations}>重新读取项目关系</Button>
            </div>
          )}
          <RelationVisibilityControls
            value={relations}
            onChange={onRelations}
            available={relationAvailability}
          />
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
              {arranging ? <X /> : <Wand2 />}
              {arranging ? "取消整理" : "自动整理"}
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
