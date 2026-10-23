"use client";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { useState } from "react";
import { Plus, RefreshCw, Play, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CanvasToolbarToggle } from "../workspace/toolbar-toggle";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { shotVideo } from "@/lib/canvas/resource-media";
import { resourceNode } from "@/lib/canvas/placement";
import { resourceKey, resourceTitle } from "@/lib/canvas/resource-presentation";
import { CHILD_TYPES } from "@/lib/canvas/resource-fields";
import { canvasTypeNames } from "@/lib/canvas/labels";
import type { CanvasResource } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { useCanvasCollection } from "../forms/use-collection";
import { useCanvasBusinessCommand } from "../forms/use-business-command";
import { CanvasBusinessCommandState } from "../forms/business-command-state";
import { CanvasCreateResourceDialog } from "../forms/create-resource-dialog";
import { CanvasResourceEditor } from "../forms/resource-editor";
import { CanvasShotBatchDialog } from "../forms/shot-batch-dialog";
import { CanvasSequencePreview } from "../preview/sequence-preview";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";
import { useCanvasLibrary } from "../library/use-library";
import { CanvasCollectionRows, type CollectionMode } from "./collection-rows";

export function CanvasResourceExplorer({
  resource,
  onClose,
}: {
  resource: CanvasResource;
  onClose: () => void;
}) {
  const [path, setPath] = useState([resource]);
  const current = path[path.length - 1];
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex h-[min(850px,calc(100vh-2rem))] sm:max-w-6xl flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{resourceTitle(current)}</DialogTitle>
          <DialogDescription>
            阅读与编辑项目内容，也可以选择内容加入画布。
          </DialogDescription>
        </DialogHeader>
        <nav className="flex shrink-0 flex-wrap gap-2" aria-label="内容层级">
          {path.map((item, index) => (
            <Button
              key={resourceKey(item)}
              size="xs"
              variant={index === path.length - 1 ? "secondary" : "ghost"}
              onClick={() => setPath(path.slice(0, index + 1))}
            >
              {resourceTitle(item)}
            </Button>
          ))}
        </nav>
        <ExplorerContent
          key={resourceKey(current)}
          resource={current}
          onOpen={(item) => setPath([...path, item])}
        />
      </DialogContent>
    </Dialog>
  );
}

function ExplorerContent({
  resource,
  onOpen,
}: {
  resource: CanvasResource;
  onOpen: (item: CanvasResource) => void;
}) {
  const ref = {
    type: resource.header.resourceType,
    id: resource.header.resourceId,
    version: resource.header.version,
  };
  const collection = useCanvasCollection(ref, 3);
  const library = useCanvasLibrary();
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => !!state.pending);
  const [mode, setMode] = useState<CollectionMode>("table");
  const [showDescendants, setShowDescendants] = useState(false);
  const [selection, setSelection] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [onlyMissing, setOnlyMissing] = useState(false);
  const [creating, setCreating] = useState(false);
  const [batch, setBatch] = useState<CanvasResource[] | null>(null);
  const [sequence, setSequence] = useState<CanvasResource[] | null>(null);
  const [preview, setPreview] = useState<CanvasMediaPreview | null>(null);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const command = useCanvasBusinessCommand(
    `children:${resourceKey(resource)}`,
    () => {
      collection.refresh();
      setSelection([]);
    },
  );
  const root = collection.result?.root;
  const all = collection.result?.descendants ?? [];
  const direct = all.filter(
    (entry) => entry.parentType === ref.type && entry.parentId === ref.id,
  );
  const selected = all.filter((item) => selection.includes(resourceKey(item)));
  const shots = (selected.length ? selected : all).filter(
    (item) => item.header.resourceType === "shot",
  );
  const editable =
    !!root?.canEdit &&
    snapshot.permissions.canEdit &&
    !snapshot.document.archived &&
    !pending;
  const disabled =
    !editable ||
    collection.loading ||
    command.busy ||
    !!command.pending ||
    command.invalid;
  const child = CHILD_TYPES[ref.type];
  const isStoryboard = [
    "storyboard",
    "storyboardEpisode",
    "storyboardScene",
  ].includes(ref.type);
  const canReorder = [
    "scriptEpisode",
    "storyboard",
    "storyboardEpisode",
    "storyboardScene",
  ].includes(ref.type);
  const visible = (showDescendants || mode === "read" ? all : direct).filter(
    (entry) =>
      resourceTitle(entry).toLowerCase().includes(search.toLowerCase()) &&
      (!onlyMissing ||
        (entry.header.resourceType === "shot" &&
          (!entry.data.firstFrameImageUrl || !shotVideo(entry.data)))),
  );
  const place = async (resources: CanvasResource[]) => {
    try {
      if (!resources.length || resources.length > 500)
        throw new Error("请一次选择 1 至 500 项内容加入画布");
      const state = store.getState();
      const nodes = resources.map((item, index) =>
        resourceNode(
          state.catalog!,
          {
            type: item.header.resourceType,
            id: item.header.resourceId,
            version: item.header.version,
          },
          resourceTitle(item),
          { x: 64 + (index % 4) * 360, y: 64 + Math.floor(index / 4) * 280 },
        ),
      );
      await state.edit(
        "展开内容引用",
        nodes.map((node) => ({ type: "canvas.node.create", node })),
      );
      store.getState().focus(nodes.map((node) => node.id));
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };
  const move = (index: number, direction: -1 | 1) => {
    if (!root || disabled) return;
    const ids = direct.map((item) => item.header.resourceId);
    [ids[index], ids[index + direction]] = [ids[index + direction], ids[index]];
    const body = {
      resourceId: root.header.resourceId,
      expectedVersion: root.header.version,
      mode: "REORDER",
    };
    if (ref.type === "scriptEpisode")
      command.run("script-scenes", { ...body, scenes: [], sceneIds: ids });
    else
      command.run("storyboard-children", {
        ...body,
        resourceType: ref.type,
        children: [],
        childIds: ids,
      });
  };
  return (
    <>
      <div className="flex shrink-0 flex-wrap items-center gap-2" role="group" aria-label="项目内容操作">
        {child && <Button
          variant={editing ? "secondary" : "outline"}
          size="default"
          onClick={() => setEditing(!editing)}
        >
          {editing ? "返回内容列表" : "查看详情"}
        </Button>}
        {child && (
          <Button
            size="default"
            disabled={disabled}
            onClick={() => setCreating(true)}
          >
            <Plus />
            新建{canvasTypeNames[child.type]}
          </Button>
        )}
        <Button
          variant="ghost"
          size="icon"
          title={child ? "刷新结构" : "刷新内容"}
          disabled={collection.loading}
          onClick={collection.refresh}
        >
          <RefreshCw />
        </Button>
        {child && <CanvasSegmentedControl
          label="项目内容视图"
          className="ml-auto"
          value={mode}
          options={[
            { value: "table", label: "表格" },
            { value: "cards", label: "卡片" },
            { value: "read", label: "连续阅读" },
          ]}
          onChange={(value) => {
            setMode(value);
            setEditing(false);
          }}
        />}
      </div>
      <CanvasBusinessCommandState
        command={command}
        onReload={collection.refresh}
      />
      {(error || collection.error || library.error) && (
        <p role="alert" className="shrink-0 text-sm text-destructive">
          {error ?? collection.error ?? library.error}
        </p>
      )}
      {collection.loading && (
        <p role="status" className="text-xs text-muted-foreground">
          正在读取最新结构…
        </p>
      )}
      {root && (editing || !child) ? (
        <OverlayScrollArea className="min-h-0 flex-1">
          <CanvasResourceEditor resource={root} onPreview={setPreview} />
        </OverlayScrollArea>
      ) : (
        <>
          <div className="flex shrink-0 flex-wrap items-center gap-2" role="group" aria-label="项目内容筛选">
            <Input
              aria-label="筛选项目内容"
              placeholder="搜索内容"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              className="w-56"
            />
            <CanvasToolbarToggle label="包含下级" checked={showDescendants} onCheckedChange={setShowDescendants} />
            {isStoryboard && (
              <CanvasToolbarToggle label="缺少首帧或视频" checked={onlyMissing} onCheckedChange={setOnlyMissing} />
            )}
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2" role="group" aria-label="所选内容操作">
            <Button
              variant="ghost"
              size="default"
              onClick={() => setSelection(visible.map(resourceKey))}
            >
              全选当前列表
            </Button>
            <Button variant="ghost" size="default" onClick={() => setSelection([])}>
              取消选择
            </Button>
            <span className="mr-auto text-xs text-muted-foreground">
              已选 {selected.length} / {all.length} 项内容
            </span>
            <Button
              variant="outline"
              size="default"
              disabled={disabled || !selected.length}
              onClick={() => void place(selected)}
            >
              <ArrowUpRight />
              加入画布
            </Button>
            {isStoryboard && (
              <>
                <Button
                  variant="outline"
                  size="default"
                  disabled={!shots.length}
                  onClick={() => setSequence(shots)}
                >
                  <Play />
                  镜头预览
                </Button>
                <Button
                  variant="outline"
                  size="default"
                  disabled={
                    disabled ||
                    !selected.length ||
                    selected.some((item) => item.header.resourceType !== "shot")
                  }
                  onClick={() => setBatch(selected)}
                >
                  批量修改
                </Button>
              </>
            )}
          </div>
          <OverlayScrollArea className="min-h-0 flex-1">
            <CanvasCollectionRows
              emptyMessage={collection.loading ? "正在读取内容…" : collection.error ? "读取失败，请点击上方刷新按钮重试。" : search.trim() || onlyMissing ? "没有符合当前筛选条件的内容，请调整或清空筛选。" : "暂无内容，可在上方新建。"}
              entries={visible}
              mode={mode}
              selected={selection}
              onSelect={(key, checked) =>
                setSelection((current) =>
                  checked
                    ? [...new Set([...current, key])]
                    : current.filter((id) => id !== key),
                )
              }
              onOpen={onOpen}
              onPlace={(items) => void place(items)}
              onPreview={setPreview}
              disabled={disabled}
              library={library.entries}
              onMove={
                canReorder &&
                !showDescendants &&
                mode !== "read" &&
                !search &&
                !onlyMissing
                  ? move
                  : undefined
              }
            />
          </OverlayScrollArea>
        </>
      )}
      {creating && root && child && (
        <CanvasCreateResourceDialog
          parent={root}
          initialType={child.type}
          onClose={() => {
            setCreating(false);
            collection.refresh();
          }}
        />
      )}
      {batch && (
        <CanvasShotBatchDialog
          shots={batch}
          onClose={() => setBatch(null)}
          onSaved={collection.refresh}
        />
      )}
      {sequence && (
        <CanvasSequencePreview
          shots={sequence}
          onClose={() => setSequence(null)}
        />
      )}
      {preview && (
        <CanvasMediaPreviewDialog
          media={preview}
          onClose={() => setPreview(null)}
        />
      )}
    </>
  );
}
