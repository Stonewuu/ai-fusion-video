import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

/** Keep the shared controls' outer focus rings inside the scroll boundary. */
export function DialogScrollBody({
  className,
  ...props
}: ComponentProps<"div">) {
  return (
    <div
      {...props}
      data-slot="dialog-scroll-body"
      className={cn("min-h-0 min-w-0 overflow-y-auto scroll-p-1 p-1", className)}
    />
  );
}
