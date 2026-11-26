import type { ComponentProps } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export function SearchInput({
  containerClassName,
  className,
  ...props
}: ComponentProps<typeof Input> & { containerClassName?: string }) {
  return (
    <div className={cn("relative min-w-0", containerClassName)}>
      <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input {...props} type="search" className={cn("pl-9", className)} />
    </div>
  );
}
