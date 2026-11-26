import type { ComponentProps } from "react";
import { Input } from "@/components/ui/input";

export function AuthInput(props: Omit<ComponentProps<typeof Input>, "className">) {
  return <Input {...props} className="h-12 px-5 text-base md:text-base" />;
}
