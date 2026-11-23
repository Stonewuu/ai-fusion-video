"use client";

import { memo, useCallback, useRef, useState } from "react";
import { CircleHelp, FolderOpen, Plus, RefreshCw, Search } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasInput } from "../ui/canvas-field";
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
      setOpening(`${entry.ref.type}:${entry.ref.id}`);
      setActiveKey(`${entry.ref.type}:${entry.ref.id}`);
      setError(null);
      try {
        const value = await canvasApi.resource(
          store.getState().canvasId,
          entry.ref.type,
          entry.ref.id,
        );
        if (openRequest.current === request) {
          setOpened(value);
        }
      } catch (failure) {
        if (openRequest.current === request) {
          setError(getApiErrorMessage(failure));
        }
      } finally {
        if (openRequest.current === request) {
          setOpening(null);
        }
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
    <section
      className="flex flex-col gap-3 p-3.5"
      aria-label="项目内容与节点目录"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-[13px] font-medium">
          {snapshot.document.projectId ? "项目内容" : "独立资产"}
        </h2>
        <div className="flex items-center gap-1">
          <Popover>
            <PopoverTrigger
              render={
                <CanvasButton
                  variant="ghost"
                  size="icon-sm"
                  aria-label="项目目录使用说明"
                  title="使用说明"
                />
              }
            >
              <CircleHelp className="text-muted-foreground" />
            </PopoverTrigger>
            <PopoverContent
              align="start"
              className="w-64 max-w-[calc(100vw-2rem)] rounded-[16px] border border-foreground/[0.1] shadow-xl dark:border-white/[0.12]"
            >
              <PopoverTitle>使用项目目录</PopoverTitle>
              <p className="text-xs leading-relaxed text-muted-foreground">
                点击名称查看内容；拖入画布或点击行末加号添加引用，点击分类可展开或折叠。
              </p>
            </PopoverContent>
          </Popover>
          <CanvasIconButton
            size="icon-sm"
            label="刷新内容目录"
            disabled={library.loading}
            onClick={library.refresh}
          >
            <RefreshCw />
          </CanvasIconButton>
          <CanvasButton
            size="sm"
            variant="secondary"
            disabled={!editable || pending}
            onClick={onCreate}
          >
            <Plus />
            新建
          </CanvasButton>
        </div>
      </div>
      <div className="relative p-0.5">
        <Search
          className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
        <CanvasInput
          name="canvas-content-search"
          placeholder="搜索名称或类型"
          aria-label="搜索内容与节点"
          className="pl-8"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      {(error || library.error) && (
        <p
          role="alert"
          className="rounded-[10px] border border-destructive/20 bg-destructive/5 px-2.5 py-2 text-xs text-destructive"
        >
          {error ?? library.error}
        </p>
      )}
      {library.loading && (
        <p role="status" className="text-xs text-muted-foreground">
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
        <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-6 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
          <span className="grid size-8 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
            <FolderOpen className="size-4" aria-hidden />
          </span>
          <p className="text-xs text-muted-foreground">
            {snapshot.document.projectId
              ? "项目里还没有剧本、分镜或资产"
              : "还没有可复用的独立资产"}
          </p>
          {editable && (
            <CanvasButton
              size="sm"
              variant="secondary"
              disabled={pending}
              onClick={onCreate}
            >
              <Plus />
              新建内容
            </CanvasButton>
          )}
        </div>
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
