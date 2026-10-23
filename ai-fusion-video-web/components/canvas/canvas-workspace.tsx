"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { mediaNode } from "@/lib/canvas/placement";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { ManagedMedia } from "@/lib/api/media";
import {
  CanvasProvider,
  useCanvas,
  useCanvasStoreApi,
} from "./canvas-provider";
import { canvasLocation } from "@/lib/canvas/location";
import { CanvasFlow } from "./editor/canvas-flow";
import { CanvasContentPanel } from "./library/content-panel";
import { CanvasMediaPicker } from "./library/media-picker";
import { CanvasEdgeInspector } from "./inspector/edge-inspector";
import { CanvasNodeInspector } from "./inspector/node-inspector";
import { CanvasSelectionInspector } from "./inspector/selection-inspector";
import { CanvasMediaPreviewDialog } from "./preview/media-preview";
import { CanvasCreateResourceDialog } from "./forms/create-resource-dialog";
import { CanvasCommentsPanel } from "./review/comments-panel";
import { CanvasHistoryPanel } from "./review/history-panel";
import { CanvasWorkspaceHeader } from "./workspace/canvas-header";
import { CanvasWorkspacePanels } from "./workspace/canvas-panels";
import { CanvasRunsProvider, useCanvasRuns } from "./runs/canvas-runs-provider";
import { CanvasRunPanel } from "./runs/run-panel";
import { CanvasMediaImportsProvider } from "./imports/media-imports-provider";
import { CanvasImportDetail } from "./imports/import-detail";
import { CanvasImportPanel } from "./imports/import-panel";
import {
  CanvasRunPlanDialog,
  type CanvasRunDraft,
} from "./runs/run-plan-dialog";
import type {
  CanvasLeftPanel,
  CanvasRightPanel,
  CanvasMobilePanel,
  CanvasPanel,
} from "./workspace/panel-navigation";

export function CanvasWorkspace({
  canvasId,
  projectId = null,
}: {
  canvasId: string;
  projectId?: number | null;
}) {
  const user = useAuthStore((state) => state.user);
  const fetchUser = useAuthStore((state) => state.fetchUserInfo);
  const resourceScope = useResourceChangeStore((state) => state.scope);
  if (!user)
    return (
      <div className="flex min-h-0 grow basis-0 items-center justify-center">
        <Button variant="outline" onClick={() => void fetchUser()}>
          读取登录信息
        </Button>
      </div>
    );
  return (
    <CanvasProvider key={user.id + ":" + canvasId} canvasId={canvasId}>
      <CanvasRunsProvider
        key={`${user.id}:${resourceScope}:${canvasId}`}
        userId={user.id}
      >
        <CanvasMediaImportsProvider
          key={`${user.id}:${resourceScope}:${canvasId}`}
          userId={user.id}
        >
          <WorkspaceContent expectedProjectId={projectId} userId={user.id} />
          <CanvasImportDetail />
        </CanvasMediaImportsProvider>
      </CanvasRunsProvider>
    </CanvasProvider>
  );
}

function WorkspaceContent({
  expectedProjectId,
  userId,
}: {
  expectedProjectId: number | null;
  userId: number;
}) {
  const router = useRouter();
  const store = useCanvasStoreApi();
  const runtime = useCanvasRuns();
  const runtimeRef = useRef(runtime);
  useEffect(() => { runtimeRef.current = runtime; }, [runtime]);
  const snapshot = useCanvas((state) => state.snapshot);
  const loading = useCanvas((state) => state.loading);
  const error = useCanvas((state) => state.error);
  const catalog = useCanvas((state) => state.catalog);
  const pending = useCanvas((state) => state.pending);
  const selection = useCanvas((state) => state.selection);
  const edgeSelection = useCanvas((state) => state.edgeSelection);
  const [left, setLeft] = useState<CanvasLeftPanel | null>("content");
  const [right, setRight] = useState<CanvasRightPanel | null>(null);
  const [mobile, setMobile] = useState<CanvasMobilePanel>(null);
  const [visited, setVisited] = useState<CanvasPanel[]>(["content"]);
  const [preview, setPreview] = useState<{
    media: CanvasMediaPreview;
    returnFocus?: HTMLElement;
  } | null>(null);
  const [creating, setCreating] = useState(false);
  const openCreate = useCallback(() => setCreating(true), []);
  const [runDraft, setRunDraft] = useState<CanvasRunDraft | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const openRight = useCallback((panel: CanvasRightPanel) => {
    setRight(panel);
    setMobile("right");
    setVisited((current) =>
      current.includes(panel) ? current : [...current, panel],
    );
  }, []);
  const inspect = useCallback(
    (id: string) => {
      store.getState().select([id]);
      openRight("properties");
    },
    [store, openRight],
  );
  const inspectEdge = useCallback(
    (id: string) => {
      store.getState().select([], [id]);
      openRight("properties");
    },
    [store, openRight],
  );
  const showPreview = useCallback(
    (media: CanvasMediaPreview, returnFocus?: HTMLElement) =>
      setPreview({ media, returnFocus }),
    [],
  );
  const runNode = useCallback(async (nodeId: string) => {
    const result = await runtimeRef.current.startNode(nodeId);
    if (result.review) setRunDraft(result.review);
    return result;
  }, []);
  const showNodeRun = useCallback((runId?: string) => {
    if (runId) runtimeRef.current.chooseRun(runId);
    openRight("runs");
  }, [openRight]);
  const previewRun = (requested?: CanvasRunDraft) => {
    const draft: CanvasRunDraft = requested ?? {
      nodeIds: selection,
      mode: selection.length ? "DEPENDENCIES" : "ALL",
      reuse: {},
    };
    setRunDraft(draft);
    void runtime.preview(draft.mode, draft.nodeIds, draft.reuse);
  };
  const addMedia = async (media: ManagedMedia) => {
    if (!snapshot) return;
    try {
      const node = mediaNode(media, {
        x: 64 + (snapshot.graph.nodes.length % 4) * 360,
        y: 64 + Math.floor(snapshot.graph.nodes.length / 4) * 320,
      });
      await store
        .getState()
        .edit("添加媒体节点", [{ type: "canvas.node.create", node }]);
      store.getState().focus([node.id]);
      setOperationError(null);
      setMobile(null);
    } catch (failure) {
      setOperationError(getApiErrorMessage(failure));
    }
  };
  if (!snapshot || !catalog)
    return (
      <div className="flex min-h-0 grow basis-0 flex-col items-center justify-center gap-3 px-4 pb-20 md:pb-3">
        {loading && (
          <Loader2 className="size-6 animate-spin motion-reduce:animate-none" />
        )}
        <p
          role={error ? "alert" : "status"}
          className="text-sm text-muted-foreground"
        >
          {error ?? "正在打开画布…"}
        </p>
        <Button
          variant="outline"
          onClick={() =>
            router.push(
              expectedProjectId
                ? "/projects/" + expectedProjectId + "/canvases"
                : "/canvases",
            )
          }
        >
          返回画布列表
        </Button>
      </div>
    );
  if (snapshot.document.projectId !== expectedProjectId)
    return (
      <div className="space-y-4 p-6">
        <p>此画布属于其他工作区。</p>
        <Button
          onClick={() => router.replace(canvasLocation(snapshot.document))}
        >
          打开对应工作区
        </Button>
      </div>
    );
  const editable =
    snapshot.permissions.canEdit && !snapshot.document.archived && !pending;
  const panels = {
    content: <CanvasContentPanel onCreate={openCreate} />,
    media: (
      <div className="space-y-3 p-4">
        <h2 className="text-sm font-semibold">媒体素材</h2>
        <CanvasImportPanel />
        <p className="text-xs text-muted-foreground">
          拖到画布，或点击素材放入。支持图片、视频、音频和文档。
        </p>
        <CanvasMediaPicker
          disabled={!editable}
          onSelect={(media) => void addMedia(media)}
        />
      </div>
    ),
    comments: <CanvasCommentsPanel />,
    history: <CanvasHistoryPanel />,
    runs: <CanvasRunPanel onPreview={previewRun} />,
    properties:
      edgeSelection.length === 1 && selection.length === 0 ? (
        <CanvasEdgeInspector edgeId={edgeSelection[0]} />
      ) : selection.length === 1 && edgeSelection.length === 0 ? (
        <CanvasNodeInspector nodeId={selection[0]} onPreview={showPreview} />
      ) : (
        <CanvasSelectionInspector
          onInspect={inspect}
          onInspectEdge={inspectEdge}
        />
      ),
  };
  return (
    <div
      className="flex min-h-0 grow basis-0 flex-col px-3 pb-20 md:pb-3"
      data-canvas-workspace={snapshot.document.id}
    >
      <div className="flex min-h-0 grow basis-0 flex-col overflow-hidden rounded-xl border border-border/30 bg-card/50">
        <CanvasWorkspaceHeader
          onRun={() => previewRun()}
          onOpenRuns={() => openRight("runs")}
        />
        {operationError && (
          <p
            role="alert"
            className="shrink-0 border-b border-border/20 px-4 py-2 text-sm text-destructive"
          >
            {operationError}
          </p>
        )}
        <CanvasWorkspacePanels
          left={left}
          right={right}
          mobile={mobile}
          visited={visited}
          content={panels}
          onLeft={(value) => {
            setLeft(value);
            setMobile("left");
            setVisited((current) =>
              current.includes(value) ? current : [...current, value],
            );
          }}
          onRight={openRight}
          onCloseLeft={() => {
            setLeft(null);
            setMobile(null);
          }}
          onCloseRight={() => {
            setRight(null);
            setMobile(null);
          }}
        >
          <CanvasFlow
            onInspect={inspect}
            onInspectEdge={inspectEdge}
            onPreview={showPreview}
            onRunNode={runNode}
            onShowNodeRun={showNodeRun}
            viewKey={"canvas-view:" + userId + ":" + snapshot.document.id}
          />
        </CanvasWorkspacePanels>
      </div>
      {preview && (
        <CanvasMediaPreviewDialog
          media={preview.media}
          returnFocus={preview.returnFocus}
          onClose={() => setPreview(null)}
        />
      )}
      {creating && (
        <CanvasCreateResourceDialog onClose={() => setCreating(false)} />
      )}
      {runDraft && (
        <CanvasRunPlanDialog
          draft={runDraft}
          onClose={() => setRunDraft(null)}
          onStarted={() => {
            setRunDraft(null);
            openRight("runs");
          }}
        />
      )}
    </div>
  );
}
