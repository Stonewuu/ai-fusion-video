"use client";

import { useRef, useState } from "react";
import {
  Scan,
  Search,
  SlidersHorizontal,
  Square,
  SquareCheck,
  Wand2,
  type LucideIcon,
} from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { CanvasRelationAvailability, CanvasRelationScope } from "@/lib/canvas/relations";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasFloatingBar } from "../ui/canvas-surface";
import { RelationVisibilityControls } from "./relation-visibility-controls";

export function CanvasToolbarAction({
  icon: Icon,
  label,
  hint,
  shortcut,
  onClick,
  disabled,
  primary = false,
}: {
  icon: LucideIcon;
  label: string;
  hint?: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <CanvasIconButton
      variant={primary ? "primary" : "ghost"}
      label={label}
      hint={hint}
      shortcut={shortcut}
      side="bottom"
      data-canvas-add={primary || undefined}
      onClick={onClick}
      disabled={disabled}
    >
      <Icon />
    </CanvasIconButton>
  );
}

export function CanvasViewActions({
  onSearch,
  onFit,
  onArrange,
  editable,
  arrangeTitle,
}: {
  onSearch: () => void;
  onFit: () => void;
  onArrange: () => void;
  editable: boolean;
  arrangeTitle: string;
}) {
  return (
    <>
      <span className="hidden @md:inline-flex">
        <CanvasToolbarAction
          icon={Search}
          label="查找"
          hint="查找并定位节点"
          shortcut="/"
          onClick={onSearch}
        />
      </span>
      <CanvasToolbarAction
        icon={Scan}
        label="查看全部"
        shortcut="F"
        onClick={onFit}
      />
      <span className="hidden @md:inline-flex">
        <CanvasToolbarAction
          icon={Wand2}
          label="自动整理"
          hint={arrangeTitle}
          onClick={onArrange}
          disabled={!editable}
        />
      </span>
    </>
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
  arrangeTitle,
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
  arrangeTitle: string;
}) {
  const [open, setOpen] = useState(false);
  const handoffFocus = useRef(false);
  return (
    <CanvasFloatingBar className="shrink-0">
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
            <CanvasButton
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
          <CanvasButton
            variant={minimap ? "secondary" : "ghost"}
            role="checkbox"
            aria-checked={minimap}
            className="w-full justify-start"
            onClick={onMinimap}
          >
            {minimap ? <SquareCheck aria-hidden className="text-primary" /> : <Square aria-hidden />}
            小地图
          </CanvasButton>
          {relationLoading && (
            <p role="status" className="text-xs text-muted-foreground">正在读取项目关系…</p>
          )}
          {relationError && (
            <div className="flex flex-col gap-2">
              <p role="alert" className="text-xs text-destructive">{relationError}</p>
              <CanvasButton variant="secondary" onClick={onReloadRelations}>重新读取项目关系</CanvasButton>
            </div>
          )}
          <RelationVisibilityControls
            value={relations}
            onChange={onRelations}
            available={relationAvailability}
          />
          <div className="grid grid-cols-2 gap-2 border-t border-border/20 pt-4">
            <CanvasButton
              variant="secondary"
              className="justify-start"
              onClick={() => {
                handoffFocus.current = true;
                setOpen(false);
                onSearch();
              }}
            >
              <Search />
              查找
            </CanvasButton>
            <CanvasButton
              variant="secondary"
              className="justify-start"
              onClick={() => {
                setOpen(false);
                onArrange();
              }}
              disabled={!editable}
              title={arrangeTitle}
            >
              <Wand2 />
              自动整理
            </CanvasButton>
          </div>
        </PopoverContent>
      </Popover>
    </CanvasFloatingBar>
  );
}
