"use client";
import { useId } from "react";
import { ChevronDown, Workflow } from "lucide-react";
import { Popover } from "@base-ui/react/popover";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useCanvasAssistantSession } from "@/lib/canvas/assistant-session";

export function AssistantCanvasContext() {
  const binding = useCanvasAssistantSession((state) => state.binding);
  const following = useCanvasAssistantSession((state) => state.following);
  const setFollowing = useCanvasAssistantSession((state) => state.setFollowing);
  const highlight = useCanvasAssistantSession((state) => state.highlight);
  const followId = useId();
  const descriptionId = useId();
  if (!binding) return null;
  const selectionLabel = binding.selectedCount ? `已选 ${binding.selectedCount} 个节点` : "当前画布";
  return (
    <Popover.Root>
      <Popover.Trigger render={
        <Button
          variant="secondary"
          size="xs"
          className="min-w-0 max-w-full"
          aria-label={`画布引用：${binding.title}，${selectionLabel}，查看详情`}
          title="查看画布引用与自动定位设置"
        />
      }>
        <Workflow />
        <span className="min-w-0 max-w-40 truncate">{binding.title}</span>
        <span className="shrink-0 text-muted-foreground">{selectionLabel}</span>
        <ChevronDown />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner side="top" align="start" sideOffset={8} className="isolate z-[120]">
          <Popover.Popup className="flex max-h-[calc(100dvh-2rem)] w-80 max-w-[calc(100vw-2rem)] origin-(--transform-origin) flex-col gap-4 overflow-y-auto rounded-2xl border border-border/40 bg-popover p-4 text-sm text-popover-foreground shadow-xl outline-hidden duration-150 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 motion-reduce:transition-none">
            <div className="space-y-1">
              <Popover.Title className="text-sm font-semibold">画布引用</Popover.Title>
              <p className="break-words text-sm">{binding.title}</p>
              <Popover.Description className="text-xs leading-relaxed text-muted-foreground">
                {binding.selectedCount
                  ? `发送时会附带当前画布与选中的 ${binding.selectedCount} 个节点。`
                  : "发送时会附带当前画布。选中节点，可让助手围绕这些内容处理。"}
              </Popover.Description>
              {highlight?.canvasId === binding.canvasId && (
                <p role="status" className="text-xs text-primary">助手刚修改了 {highlight.nodeIds.length} 个节点</p>
              )}
            </div>
            <div className="space-y-2 border-t border-border/20 pt-3">
              <label htmlFor={followId} className="flex cursor-pointer items-center gap-3 font-medium">
                <Checkbox id={followId} checked={following} onCheckedChange={setFollowing} aria-describedby={descriptionId} />
                自动定位到助手结果
              </label>
              <p id={descriptionId} className="text-xs leading-relaxed text-muted-foreground">
                {following
                  ? "助手定位节点时，画布会自动移动到对应位置。"
                  : "画布保持当前视野，你可以自行拖动和缩放。"}
              </p>
            </div>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  );
}
