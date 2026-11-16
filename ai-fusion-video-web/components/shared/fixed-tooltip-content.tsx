"use client";

import { Tooltip } from "@base-ui/react/tooltip";
import { cn } from "@/lib/utils";

export function FixedTooltipContent({
  children,
  className,
  side = "top",
  align = "center",
  sideOffset = 4,
  ...props
}: Tooltip.Popup.Props &
  Pick<Tooltip.Positioner.Props, "side" | "align" | "sideOffset">) {
  return (
    <Tooltip.Portal>
      <Tooltip.Positioner
        positionMethod="fixed"
        side={side}
        align={align}
        sideOffset={sideOffset}
        collisionPadding={8}
        className="pointer-events-none isolate z-50 data-anchor-hidden:invisible"
      >
        <Tooltip.Popup
          {...props}
          className={cn(
            "w-fit max-w-xs origin-(--transform-origin) whitespace-normal break-words rounded-2xl bg-foreground px-3 py-1.5 text-xs text-background data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 motion-reduce:duration-0",
            className,
          )}
        >
          {children}
          <Tooltip.Arrow className="size-2.5 rotate-45 rounded-[2px] bg-foreground data-[side=top]:-bottom-1 data-[side=bottom]:-top-1 data-[side=left]:-right-1 data-[side=right]:-left-1" />
        </Tooltip.Popup>
      </Tooltip.Positioner>
    </Tooltip.Portal>
  );
}
