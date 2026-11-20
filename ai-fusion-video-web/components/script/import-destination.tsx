import { AlertCircle } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const modes = [
  { value: "APPEND", label: "追加到现有分集" },
  { value: "REPLACE", label: "替换全部分集" },
];

export function ImportDestination({
  count,
  mode,
  disabled,
  onMode,
}: {
  count: number;
  mode: "APPEND" | "REPLACE";
  disabled: boolean;
  onMode: (mode: "APPEND" | "REPLACE") => void;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <label htmlFor="source-import-mode" className="text-sm font-medium">
          导入方式
        </label>
        <Select
          items={modes}
          value={mode}
          disabled={disabled}
          onValueChange={(value) => {
            if (value === "APPEND" || value === "REPLACE") {
              onMode(value);
            }
          }}
        >
          <SelectTrigger id="source-import-mode" className="w-44">
            <SelectValue placeholder="选择导入方式" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {modes.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
        <span className="text-xs text-muted-foreground">
          现有 {count.toLocaleString()} 集
        </span>
      </div>
      {mode === "REPLACE" && (
        <p className="flex items-start gap-1.5 text-xs text-destructive">
          <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
          现有分集与场次将被替换，已有分镜保留。
        </p>
      )}
    </div>
  );
}
