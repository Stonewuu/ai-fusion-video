"use client";
import { canvasUploadFile } from "@/lib/canvas/files";
import {
  useEffect,
  useRef,
  useState,
  type ClipboardEvent,
  type DragEvent,
} from "react";
import { toast } from "sonner";
import { uploadMedia, type ManagedMedia } from "@/lib/api/media";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { copyNodes } from "@/lib/canvas/graph";
import { mediaNode, resourceNode } from "@/lib/canvas/placement";
import {
  CANVAS_CLIPBOARD,
  decodeCanvasClipboard,
  encodeCanvasClipboard,
} from "@/lib/canvas/clipboard";
import { isEditableTarget } from "@/lib/canvas/layout-actions";
import type { CanvasNode, CanvasOperation } from "@/lib/canvas/types";
import type { CanvasStore } from "@/lib/store/canvas-store";

type Point = { x: number; y: number };
type Upload = { file: File; id: string; media?: ManagedMedia };
type Batch = { items: Upload[]; position: Point };

export function useCanvasTransfer(
  store: CanvasStore,
  editable: boolean,
  position: (point?: Point) => Point,
) {
  const [batch, setBatch] = useState<Batch | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const place = async (operations: CanvasOperation[]) => {
    if (!operations.length) return;
    await store.getState().edit("放置导入内容", operations);
    store
      .getState()
      .select(
        operations.flatMap((operation) =>
          operation.type === "canvas.node.create" ? [operation.node.id] : [],
        ),
      );
  };
  const upload = async (work: Batch) => {
    if (controller.current) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError(null);
    try {
      const snapshot = store.getState().snapshot!;
      for (const item of work.items) {
        if (abort.signal.aborted) return;
        if (item.media) continue;
        item.media = await uploadMedia(canvasUploadFile(item.file), {
          requestId: item.id,
          canvasId: snapshot.document.id,
          projectId: snapshot.document.projectId ?? undefined,
          signal: abort.signal,
        });
        if (!abort.signal.aborted)
          setBatch({ ...work, items: [...work.items] });
      }
      if (abort.signal.aborted) return;
      const nodes = work.items.map((item, index) =>
        mediaNode(item.media!, {
          x: work.position.x + (index % 3) * 340,
          y: work.position.y + Math.floor(index / 3) * 280,
        }),
      );
      // 图保存从这里交给统一命令状态；失败时用户重试同一图请求，不重新上传或再造节点。
      setBatch(null);
      await place(nodes.map((node) => ({ type: "canvas.node.create", node })));
    } catch (failure) {
      if (!abort.signal.aborted) setError(getApiErrorMessage(failure));
    } finally {
      if (controller.current === abort) controller.current = null;
      if (!abort.signal.aborted) setBusy(false);
    }
  };
  const files = (input: FileList, at: Point) => {
    if (batch || busy) throw new Error("请先处理正在导入的素材");
    if (input.length > 20) throw new Error("一次最多导入 20 个文件");
    const work = {
      items: Array.from(input).map((file) => ({
        file,
        id: crypto.randomUUID(),
      })),
      position: at,
    };
    setBatch(work);
    void upload(work);
  };
  const text = async (content: string, at: Point) => {
    if (!content.trim()) return;
    if (content.length > 128000)
      throw new Error("文本超过 128000 字符，请作为文件导入");
    const node: CanvasNode = {
      id: crypto.randomUUID(),
      kind: "text",
      schemaVersion: 1,
      title: content.trim().split(/\r?\n/)[0].slice(0, 60),
      config: { content },
      configVersion: 0,
      layoutVersion: 0,
      resource: null,
      layout: {
        ...at,
        width: 300,
        height: null,
        parentId: null,
        locked: false,
        collapsed: false,
      },
    };
    await place([{ type: "canvas.node.create", node }]);
  };
  const onPaste = (event: ClipboardEvent<HTMLDivElement>) => {
    if (!editable || isEditableTarget(event.target)) return;
    event.preventDefault();
    const data = event.clipboardData;
    try {
      if (data.files.length) {
        files(data.files, position());
        return;
      }
      const raw = data.getData(CANVAS_CLIPBOARD) || data.getData("text/plain");
      if (
        data.types.includes(CANVAS_CLIPBOARD) ||
        raw.startsWith('{"format":"afv-canvas-1",')
      ) {
        const graph = decodeCanvasClipboard(raw);
        const top = graph.nodes.filter((node) => !node.layout.parentId);
        const at = position();
        const offset = {
          x: at.x - Math.min(...top.map((node) => node.layout.x)),
          y: at.y - Math.min(...top.map((node) => node.layout.y)),
        };
        void place(
          copyNodes(
            graph,
            graph.nodes.map((node) => node.id),
            offset,
          ),
        ).catch((failure) => toast.error(getApiErrorMessage(failure)));
      } else
        void text(raw, position()).catch((failure) =>
          toast.error(getApiErrorMessage(failure)),
        );
    } catch (failure) {
      toast.error(getApiErrorMessage(failure));
    }
  };
  const onCopy = (event: ClipboardEvent<HTMLDivElement>) => {
    const state = store.getState();
    if (
      isEditableTarget(event.target) ||
      !state.selection.length ||
      !state.snapshot
    )
      return;
    try {
      const content = encodeCanvasClipboard(
        state.snapshot.graph,
        state.selection,
      );
      event.clipboardData.setData(CANVAS_CLIPBOARD, content);
      event.clipboardData.setData("text/plain", content);
      event.preventDefault();
      toast.success("已复制选中节点，可粘贴到有权限的画布");
    } catch (failure) {
      toast.error(getApiErrorMessage(failure));
    }
  };
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    if (!editable || isEditableTarget(event.target)) return;
    event.preventDefault();
    const at = position({ x: event.clientX, y: event.clientY });
    try {
      if (event.dataTransfer.files.length) {
        files(event.dataTransfer.files, at);
        return;
      }
      const rawResource = event.dataTransfer.getData(
        "application/x-fusion-resource",
      );
      const rawMedia = event.dataTransfer.getData("application/x-fusion-media");
      let node: CanvasNode;
      if (rawResource) {
        const entry = JSON.parse(rawResource);
        if (
          !entry.ref ||
          typeof entry.title !== "string" ||
          !Number.isSafeInteger(entry.ref.id) ||
          !Number.isSafeInteger(entry.ref.version)
        )
          throw new Error("拖入的业务引用格式无效");
        node = resourceNode(
          store.getState().catalog!,
          entry.ref,
          entry.title,
          at,
        );
      } else if (rawMedia) node = mediaNode(JSON.parse(rawMedia), at);
      else {
        void text(event.dataTransfer.getData("text/plain"), at).catch(
          (failure) => toast.error(getApiErrorMessage(failure)),
        );
        return;
      }
      void place([{ type: "canvas.node.create", node }]).catch((failure) =>
        toast.error(getApiErrorMessage(failure)),
      );
    } catch (failure) {
      toast.error(getApiErrorMessage(failure));
    }
  };
  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (!editable || isEditableTarget(event.target)) return;
    if (
      event.dataTransfer.types.some((type) =>
        [
          "Files",
          "application/x-fusion-resource",
          "application/x-fusion-media",
          "text/plain",
        ].includes(type),
      )
    ) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    }
  };
  return {
    onCopy,
    onPaste,
    onDrop,
    onDragOver,
    batch,
    busy,
    error,
    retry: () => {
      if (batch && !busy) void upload(batch);
    },
    cancel: () => {
      controller.current?.abort();
      controller.current = null;
      setBusy(false);
      setBatch(null);
      setError(null);
    },
  };
}
