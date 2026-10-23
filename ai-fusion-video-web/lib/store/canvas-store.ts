import { createStore } from "zustand/vanilla";
import { canvasApi, type CanvasPlacementCommand } from "@/lib/api/canvas";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { applyCanvasChanges, reconcileCanvasGraph } from "@/lib/canvas/graph";
import type {
  CanvasCatalog,
  CanvasEdit,
  CanvasEvent,
  CanvasOperation,
  CanvasReceipt,
  CanvasSnapshot,
} from "@/lib/canvas/types";

type Pending =
  | { kind: "edit"; label: string; command: CanvasEdit }
  | { kind: "resourceCreate"; label: string; command: CanvasPlacementCommand }
  | {
      kind: "undo" | "redo" | "historyUndo";
      label: string;
      commandId: string;
      requestId: string;
    };
export interface CanvasState {
  canvasId: string;
  snapshot: CanvasSnapshot | null;
  catalog: CanvasCatalog | null;
  loading: boolean;
  error: string | null;
  pending: Pending | null;
  saving: boolean;
  saveError: string | null;
  sync: "connecting" | "connected" | "retrying" | "denied";
  syncError: string | null;
  selection: string[];
  edgeSelection: string[];
  focusRequest: { ids: string[]; sequence: number } | null;
  contentRevision: number;
  undoStack: string[];
  redoStack: string[];
  load: (signal?: AbortSignal) => Promise<void>;
  reauthorize: () => void;
  refreshCatalog: () => Promise<void>;
  receive: (event: CanvasEvent) => Promise<void>;
  rememberCommand: (receipt: CanvasReceipt) => void;
  edit: (
    label: string,
    operations: CanvasOperation[],
  ) => Promise<CanvasReceipt>;
  createResource: (command: CanvasPlacementCommand) => Promise<CanvasReceipt>;
  undo: () => Promise<CanvasReceipt>;
  redo: () => Promise<CanvasReceipt>;
  undoHistory: (commandId: string) => Promise<CanvasReceipt>;
  retry: () => Promise<CanvasReceipt>;
  discardPending: () => Promise<void>;
  select: (nodes: string[], edges?: string[]) => void;
  focus: (ids: string[]) => void;
}

export function createCanvasStore(canvasId: string) {
  return createStore<CanvasState>((set, get) => {
    let scopeEpoch = 0;
    const commandRevisions = new Map<string, number>();
    const orderedCommands = (ids: string[]) =>
      [...new Set(ids)]
        .sort((a, b) => commandRevisions.get(a)! - commandRevisions.get(b)!)
        .slice(-100);
    const load = async (signal?: AbortSignal) => {
      const epoch = scopeEpoch;
      const [snapshot, catalog] = await Promise.all([
        canvasApi.snapshot(canvasId, signal),
        get().catalog
          ? Promise.resolve(get().catalog!)
          : canvasApi.capabilities(),
      ]);
      if (signal?.aborted || get().sync === "denied" || epoch !== scopeEpoch)
        return;
      if (snapshot.graph.schemaVersion !== catalog.schemaVersion)
        throw new Error("画布与能力目录版本不一致，请更新客户端");
      const current = get().snapshot;
      if (!current || snapshot.lastSequence >= current.lastSequence) {
        set({
          snapshot: current
            ? {
                ...snapshot,
                graph: reconcileCanvasGraph(current.graph, snapshot.graph),
              }
            : snapshot,
          catalog,
          loading: false,
          error: null,
          selection: get().selection.filter((id) =>
            snapshot.graph.nodes.some((node) => node.id === id),
          ),
          edgeSelection: get().edgeSelection.filter((id) =>
            snapshot.graph.edges.some((edge) => edge.id === id),
          ),
        });
      }
    };
    const commit = async (pending: Pending): Promise<CanvasReceipt> => {
      const epoch = scopeEpoch;
      let accepted = false;
      set({ pending, saving: true, saveError: null });
      try {
        const receipt =
          pending.kind === "edit"
            ? await canvasApi.edit(canvasId, pending.command)
            : pending.kind === "resourceCreate"
              ? (await canvasApi.createAndPlace(canvasId, pending.command))
                  .canvas
              : await canvasApi.undo(
                  canvasId,
                  pending.commandId,
                  pending.requestId,
                );
        accepted = true;
        if (epoch !== scopeEpoch)
          throw new Error("权限范围已经变化，请在历史记录中核对刚才的操作");
        await load();
        if (epoch !== scopeEpoch || get().sync === "denied")
          throw new Error("权限范围已经变化，请在历史记录中核对刚才的操作");
        commandRevisions.set(receipt.commandId, receipt.revision);
        if (pending.kind === "undo")
          set({
            undoStack: get().undoStack.filter((id) => id !== pending.commandId),
            redoStack: [...get().redoStack, receipt.commandId],
          });
        else if (pending.kind === "redo")
          set({
            redoStack: get().redoStack.filter((id) => id !== pending.commandId),
            undoStack: orderedCommands([...get().undoStack, receipt.commandId]),
          });
        else
          set({
            undoStack: orderedCommands([...get().undoStack, receipt.commandId]),
            redoStack: [],
          });
        set({ pending: null, saving: false, saveError: null });
        return receipt;
      } catch (error) {
        if (epoch !== scopeEpoch) throw error;
        const status = getApiErrorStatus(error);
        const rejected = !accepted && [400, 401, 403, 404, 409, 422].includes(status ?? 0);
        set({ saving: false, pending: rejected ? null : pending, saveError: getApiErrorMessage(error) });
        if ([403, 404].includes(getApiErrorStatus(error) ?? 0))
          set({
            snapshot: null,
            sync: "denied",
            error: getApiErrorMessage(error),
          });
        else if (rejected) void load().catch(() => undefined);
        throw error;
      }
    };
    const admitted = () => {
      if (get().pending) throw new Error("正在后台确认保存，请稍后继续编辑");
      if (
        !get().snapshot?.permissions.canEdit ||
        get().snapshot?.document.archived
      )
        throw new Error("当前画布不可编辑");
    };
    return {
      canvasId,
      snapshot: null,
      catalog: null,
      loading: true,
      error: null,
      pending: null,
      saving: false,
      saveError: null,
      sync: "connecting",
      syncError: null,
      selection: [],
      edgeSelection: [],
      focusRequest: null,
      contentRevision: 0,
      undoStack: [],
      redoStack: [],
      load,
      refreshCatalog: async () => {
        const epoch = scopeEpoch;
        const catalog = await canvasApi.capabilities();
        if (epoch === scopeEpoch && get().sync !== "denied") set({ catalog });
      },
      reauthorize: () => {
        scopeEpoch++;
        commandRevisions.clear();
        set({
          snapshot: null,
          catalog: null,
          loading: true,
          sync: "connecting",
          error: null,
          syncError: null,
          pending: null,
          saving: false,
          saveError: null,
          selection: [],
          edgeSelection: [],
          undoStack: [],
          redoStack: [],
          focusRequest: null,
        });
      },
      select: (selection, edgeSelection = []) =>
        set({ selection, edgeSelection }),
      rememberCommand: (receipt) => {
        if (receipt.canvasId !== canvasId)
          throw new Error("操作回执不属于当前画布");
        if (
          !get().snapshot?.permissions.canEdit ||
          get().sync === "denied" ||
          commandRevisions.has(receipt.commandId)
        )
          return;
        commandRevisions.set(receipt.commandId, receipt.revision);
        set({
          undoStack: orderedCommands([...get().undoStack, receipt.commandId]),
          redoStack: [],
        });
      },
      focus: (ids) =>
        set({
          selection: ids,
          edgeSelection: [],
          focusRequest: {
            ids,
            sequence: (get().focusRequest?.sequence ?? 0) + 1,
          },
        }),
      receive: async (event) => {
        if (event.canvasId !== canvasId) throw new Error("事件不属于当前画布");
        const snapshot = get().snapshot;
        if (!snapshot || event.sequence <= snapshot.lastSequence) return;
        if (
          event.sequence !== snapshot.lastSequence + 1 ||
          !["canvas.graph.changed", "canvas.graph.undone"].includes(event.type)
        ) {
          await load();
          set({ contentRevision: get().contentRevision + 1 });
          return;
        }
        const receipt = event.payload as unknown as CanvasReceipt;
        if (
          receipt.canvasId !== canvasId ||
          receipt.sequence !== event.sequence ||
          !receipt.changes ||
          receipt.revision !== snapshot.document.revision + 1
        ) {
          throw new Error("图事件与当前版本不一致");
        }
        const graph = applyCanvasChanges(snapshot.graph, receipt.changes);
        set({
          snapshot: {
            ...snapshot,
            graph,
            lastSequence: event.sequence,
            document: {
              ...snapshot.document,
              revision: receipt.revision,
              nextEventSequence: event.sequence + 1,
              updateTime: event.createdAt,
            },
          },
          selection: get().selection.filter((id) =>
            graph.nodes.some((node) => node.id === id),
          ),
          edgeSelection: get().edgeSelection.filter((id) =>
            graph.edges.some((edge) => edge.id === id),
          ),
          contentRevision: get().contentRevision + 1,
        });
      },
      edit: async (label, operations) => {
        admitted();
        if (!operations.length || operations.length > 500)
          throw new Error("单次操作应包含 1 至 500 项修改");
        return commit({
          kind: "edit",
          label,
          command: { requestId: crypto.randomUUID(), operations },
        });
      },
      createResource: async (command) => {
        admitted();
        return commit({
          kind: "resourceCreate",
          label: "新建业务内容并放置节点",
          command,
        });
      },
      undo: async () => {
        admitted();
        const commandId = get().undoStack.at(-1);
        if (!commandId) throw new Error("没有可撤销的操作");
        return commit({
          kind: "undo",
          label: "撤销",
          commandId,
          requestId: crypto.randomUUID(),
        });
      },
      redo: async () => {
        admitted();
        const commandId = get().redoStack.at(-1);
        if (!commandId) throw new Error("没有可重做的操作");
        return commit({
          kind: "redo",
          label: "重做",
          commandId,
          requestId: crypto.randomUUID(),
        });
      },
      undoHistory: async (commandId) => {
        admitted();
        return commit({
          kind: "historyUndo",
          label: "撤销历史操作",
          commandId,
          requestId: crypto.randomUUID(),
        });
      },
      retry: async () => {
        if (!get().pending || get().saving)
          throw new Error("当前没有可重试的操作");
        return commit(get().pending!);
      },
      discardPending: async () => {
        if (get().saving) throw new Error("正在保存，请等待结果");
        await load();
        set({ pending: null, saveError: null });
      },
    };
  });
}
export type CanvasStore = ReturnType<typeof createCanvasStore>;
