import Image from "next/image";
import { cn } from "@/lib/utils";

export function AssistantBrandIcon({
  className,
  loading,
}: {
  className?: string;
  loading?: "eager" | "lazy";
}) {
  return (
    <Image
      src="/brand/fusion-assistant.svg"
      alt=""
      width={64}
      height={64}
      loading={loading}
      draggable={false}
      className={cn("pointer-events-none shrink-0 select-none object-contain", className)}
    />
  );
}
