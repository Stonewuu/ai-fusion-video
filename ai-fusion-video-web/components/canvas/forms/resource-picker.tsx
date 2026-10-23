"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  loadCanvasLibrary,
  type CanvasLibraryEntry,
} from "@/lib/canvas/resource-library";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { ResourceType } from "@/lib/api/resource-events";
import { useCanvas } from "../canvas-provider";
import { cn } from "@/lib/utils";

export function CanvasResourcePicker({
  type,
  value,
  onSelect,
}: {
  type: ResourceType;
  value?: number;
  onSelect: (entry: CanvasLibraryEntry) => void;
}) {
  const projectId = useCanvas((state) => state.snapshot!.document.projectId);
  const [entries, setEntries] = useState<CanvasLibraryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  useEffect(() => {
    let active = true;
    void loadCanvasLibrary(projectId)
      .then((result) => {
        if (active) setEntries(result.filter((item) => item.ref.type === type));
      })
      .catch((failure) => {
        if (active) setError(getApiErrorMessage(failure));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [projectId, type]);
  return (
    <div className="space-y-3">
      <Input
        aria-label="搜索项目内容"
        placeholder="搜索项目内容"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {loading && (
        <p role="status" className="text-sm text-muted-foreground">
          正在读取项目内容…
        </p>
      )}
      <div className="space-y-2">
        {entries
          .filter((item) =>
            item.title.toLowerCase().includes(search.toLowerCase()),
          )
          .map((item) => (
            <Button
              key={item.ref.id}
              variant={value === item.ref.id ? "secondary" : "ghost"}
              className="w-full justify-start"
              aria-pressed={value === item.ref.id}
              onClick={() => onSelect(item)}
              title={item.title}
            >
              <span
                className={cn(
                  "truncate",
                  value === item.ref.id && "font-semibold",
                )}
              >
                {item.title}
              </span>
            </Button>
          ))}
      </div>
      {!loading && !error && !entries.length && (
        <p className="text-sm text-muted-foreground">
          暂无此类对象，可在项目内容面板创建。
        </p>
      )}
    </div>
  );
}
