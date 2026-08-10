import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

const widthClasses = {
  default: "max-w-7xl",
  settings: "max-w-[1200px]",
  full: "max-w-none",
} as const;

interface MainContentFrameProps {
  children: ReactNode;
  width?: keyof typeof widthClasses;
  fullHeight?: boolean;
  className?: string;
  contentClassName?: string;
}

export function MainContentFrame({
  children,
  width = "default",
  fullHeight = false,
  className,
  contentClassName,
}: MainContentFrameProps) {
  return (
    <div
      className={cn(
        "w-full px-5 pb-20 lg:px-6 lg:pb-3",
        fullHeight
          ? "flex min-h-0 grow basis-0 flex-col"
          : "min-h-full shrink-0",
        className,
      )}
    >
      <div
        className={cn(
          "mx-auto w-full",
          fullHeight
            ? "flex min-h-0 grow basis-0 flex-col"
            : "min-h-full",
          widthClasses[width],
          contentClassName,
        )}
      >
        {children}
      </div>
    </div>
  );
}
