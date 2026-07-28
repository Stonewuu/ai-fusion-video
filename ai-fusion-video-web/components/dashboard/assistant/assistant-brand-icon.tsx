import Image from "next/image";
import { cn } from "@/lib/utils";

export function AssistantBrandIcon({ className }: { className?: string }) {
  return (
    <Image
      src="/assistant-avatar.svg"
      alt=""
      width={64}
      height={64}
      draggable={false}
      className={cn("shrink-0 select-none object-contain", className)}
    />
  );
}
