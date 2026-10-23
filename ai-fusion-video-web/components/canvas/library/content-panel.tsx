"use client";
import { memo, useCallback, useRef, useState } from "react";
import { CircleHelp, Plus, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { canvasApi } from "@/lib/api/canvas";
import { resourceNode } from "@/lib/canvas/placement";
import type { CanvasLibraryEntry } from "@/lib/canvas/resource-library";
import type { CanvasResource } from "@/lib/canvas/types";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasResourceExplorer } from "../inspector/resource-explorer";
import { CanvasLibraryTree } from "./library-tree";
import { useCanvasLibrary } from "./use-library";
import { CanvasPlacedNodes } from "./placed-nodes";

export const CanvasContentPanel = memo(function CanvasContentPanel({
  onCreate,
}: {
  onCreate: () => void;
}) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => !!state.pending);
  const library = useCanvasLibrary();
  const [search, setSearch] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState<string | null>(null);
  const [opened, setOpened] = useState<CanvasResource | null>(null);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const openRequest = useRef(0);
  const editable = snapshot.permissions.canEdit && !snapshot.document.archived;
  const open = useCallback(
    async (entry: CanvasLibraryEntry) => {
      const request = ++openRequest.current;
      setOpening(entry.ref.type + ":" + entry.ref.id);
      setActiveKey(entry.ref.type + ":" + entry.ref.id);
      setError(null);
      try {
        const value = await canvasApi.resource(
          store.getState().canvasId,
          entry.ref.type,
          entry.ref.id,
        );
        if (openRequest.current === request) setOpened(value);
      } catch (failure) {
        if (openRequest.current === request)
          setError(getApiErrorMessage(failure));
      } finally {
        if (openRequest.current === request) setOpening(null);
      }
    },
    [store],
  );
  const add = useCallback(
    async (entry: CanvasLibraryEntry) => {
      try {
        const state = store.getState();
        const node = resourceNode(state.catalog!, entry.ref, entry.title, {
          x: 64 + (state.snapshot!.graph.nodes.length % 4) * 360,
          y: 64 + Math.floor(state.snapshot!.graph.nodes.length / 4) * 280,
        });
        await state.edit("添加内容引用", [
          { type: "canvas.node.create", node },
        ]);
        store.getState().focus([node.id]);
        setError(null);
      } catch (failure) {
        setError(getApiErrorMessage(failure));
      }
    },
    [store],
  );
  return (
    <section className="space-y-3 p-3" aria-label="项目内容与节点目录">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">
          {snapshot.document.projectId ? "项目内容" : "独立资产"}
        </h2>
        <div className="flex gap-2">
          <Popover>
            <PopoverTrigger render={<Button variant="ghost" size="icon-xs" aria-label="项目目录使用说明" />}><CircleHelp /></PopoverTrigger>
            <PopoverContent align="start" className="max-w-[calc(100vw-2rem)]">
              <PopoverTitle>使用项目目录</PopoverTitle>
              <p className="text-sm leading-relaxed text-muted-foreground">点击名称查看内容；拖入画布，或点击行末的加号添加引用。点击分类或箭头可展开、收起。</p>
            </PopoverContent>
          </Popover>
          <Button
            variant="ghost"
            size="icon-xs"
            title="刷新内容目录"
            disabled={library.loading}
            onClick={library.refresh}
          >
            <RefreshCw />
          </Button>
          <Button
            size="xs"
            variant="outline"
            disabled={!editable || pending}
            onClick={onCreate}
          >
            <Plus />
            新建
          </Button>
        </div>
      </div>
      <Input
        placeholder="搜索名称或类型"
        aria-label="搜索内容与节点"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />
      {(error || library.error) && (
        <p role="alert" className="text-sm text-destructive">
          {error ?? library.error}
        </p>
      )}
      {library.loading && (
        <p role="status" className="text-sm text-muted-foreground">
          正在读取内容…
        </p>
      )}
      <CanvasLibraryTree
        entries={library.entries}
        search={search}
        opening={opening}
        activeKey={activeKey}
        disabled={!editable}
        busy={pending}
        onOpen={open}
        onAdd={add}
      />
      {!library.loading && !library.entries.length && (
        <p className="text-sm text-muted-foreground">
          尚无内容，可在上方新建。
        </p>
      )}
      <CanvasPlacedNodes search={search} />
      {opened && (
        <CanvasResourceExplorer
          resource={opened}
          onClose={() => setOpened(null)}
        />
      )}
    </section>
  );
});
