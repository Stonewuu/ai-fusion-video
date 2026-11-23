"use client";
import { canvasAriaLabels } from "@/lib/canvas/accessibility-labels";
import { useCanvasBusinessRelations } from "./use-business-relations";
import { CanvasCompareDialog } from "../preview/compare-dialog";
import { useCanvasFlowMotion } from "./use-flow-motion";
import { useCanvasEditActions } from "./use-edit-actions";
import { useCanvasTransfer } from "./use-canvas-transfer";
import { useCanvasConnections } from "./use-connections";
import { useCanvasFlowProjection } from "./use-flow-projection";
import { useCanvasChainHighlight } from "./use-chain-highlight";
import { useCanvasEdgeMarquee } from "./use-edge-marquee";
import { useCanvasAutoLayout } from "./use-auto-layout";

import { memo, useCallback, useMemo, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  ConnectionLineType,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  SelectionMode,
  useKeyPress,
  useReactFlow,
  type ReactFlowProps,
} from "@xyflow/react";
import { useReducedMotion } from "framer-motion";
import { useTheme } from "next-themes";
import { toast } from "sonner";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { worldPosition } from "@/lib/canvas/graph";
import { nearbyNodePosition } from "@/lib/canvas/placement";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CANVAS_MAX_ZOOM, CANVAS_MIN_ZOOM } from "@/lib/canvas/viewport-limits";
import { isEditableTarget } from "@/lib/canvas/layout-actions";
import type { CanvasAddIntent, CanvasOperation } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasNodeRenderer, type CanvasFlowNode } from "./canvas-node";
import { CanvasEdgeRenderer } from "./canvas-edge";
import { CanvasToolbar } from "./canvas-toolbar";
import { CanvasSearchDialog } from "./canvas-search-dialog";
import { CanvasHelpDialog } from "./canvas-help-dialog";
import { CanvasEmptyState } from "./canvas-empty-state";
import { CanvasViewportStatus } from "./canvas-status";
import { CanvasNodeMenu, type CanvasNodeChoice } from "./node-menu";
import { useCanvasCreateNode } from "./use-create-node";
import { CanvasContextActions } from "./context-actions";
import { CanvasFlowNotices } from "./flow-notices";
import { buildCanvasSelectionActions } from "./build-context-actions";
import { CanvasSelectionActions } from "./toolbar-selection-actions";
import { useCanvasViewport } from "./use-viewport";
import { createCanvasGestureGate } from "./gesture-gate";
import { canvasNodePanelKind, hasCanvasQuickPanel } from "@/lib/canvas/generation";
import { CanvasNodePanel } from "../generation/node-panel";
import { GenerationComposer } from "../generation/generation-composer";
import { TextComposer } from "../generation/text-composer";
import { MediaNodePanel } from "../generation/media-node-panel";
import { NodeFloatingPanel } from "../generation/node-floating-panel";
import { CanvasAssistantBridge } from "../agent/canvas-assistant-bridge";
import {
  EMPTY_AGENT_NODES,
  useCanvasAssistantSession,
} from "@/lib/canvas/assistant-session";
import "@xyflow/react/dist/style.css";
import "./canvas-flow.css";

const nodeTypes = { canvas: CanvasNodeRenderer };
const edgeTypes = { canvas: CanvasEdgeRenderer };

interface Props {
  panelNodeId: string | null;
  onPanelNodeChange: (id: string | null) => void;
  onInspect: (id: string) => void;
  onInspectEdge: (id: string) => void;
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
  onRunNode: CanvasFlowNode["data"]["onRunNode"];
  onShowNodeRun: CanvasFlowNode["data"]["onShowNodeRun"];
  viewKey: string;
}

export const CanvasFlow = memo(function CanvasFlow(props: Props) {
  return (
    <ReactFlowProvider>
      <CanvasFlowEditor {...props} />
    </ReactFlowProvider>
  );
});

function CanvasFlowEditor({
  panelNodeId,
  onPanelNodeChange,
  onInspect,
  onInspectEdge,
  onPreview,
  onRunNode,
  onShowNodeRun,
  viewKey,
}: Props) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const agentChangedIds = useCanvasAssistantSession((state) =>
    state.highlight?.canvasId === snapshot.document.id
      ? state.highlight.nodeIds
      : EMPTY_AGENT_NODES,
  );
  const catalog = useCanvas((state) => state.catalog!);
  const selection = useCanvas((state) => state.selection);
  const edgeSelection = useCanvas((state) => state.edgeSelection);
  // 悬浮面板只由点击、“查看节点”或新建节点打开；拖动会选中节点，但不应弹出面板。
  const panelNode =
    selection.length === 1 &&
    edgeSelection.length === 0 &&
    selection[0] === panelNodeId
      ? snapshot.graph.nodes.find(
          (node) => node.id === panelNodeId && hasCanvasQuickPanel(node),
        )
      : undefined;
  const pending = useCanvas((state) => state.pending);
  const focusRequest = useCanvas((state) => state.focusRequest);
  const editable =
    snapshot.permissions.canEdit && !snapshot.document.archived && !pending;
  const canEdit = snapshot.permissions.canEdit && !snapshot.document.archived;
  const gestureGate = useMemo(
    () => createCanvasGestureGate(() => !!store.getState().pending),
    [store],
  );
  const flow = useReactFlow<CanvasFlowNode>();
  const reduced = useReducedMotion();
  const { resolvedTheme } = useTheme();
  const region = useRef<HTMLDivElement>(null);
  const canvasArea = useRef<HTMLDivElement>(null);
  const [intent, setIntent] = useState<CanvasAddIntent | null>(null);
  const inspect = useCallback(
    (id: string) => {
      onPanelNodeChange(id);
      onInspect(id);
    },
    [onInspect, onPanelNodeChange],
  );
  const createNode = useCanvasCreateNode(inspect);
  const [context, setContext] = useState<{
    position: { x: number; y: number };
    nodeId?: string;
    edgeId?: string;
  } | null>(null);
  const [minimap, setMinimap] = useState(false);
  const [comparing, setComparing] = useState<string[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [help, setHelp] = useState(false);
  const [pointerMode, setPointerMode] = useState<"select" | "pan">("select");
  const panKeyPressed = useKeyPress("Space");
  const edgeMarquee = useCanvasEdgeMarquee(pointerMode === "select" && !panKeyPressed);
  const { initialView, fit, saveViewport } = useCanvasViewport(
    viewKey,
    flow,
    canvasArea,
    !!reduced,
    snapshot.graph.nodes.length > 0,
    focusRequest,
  );
  const canvasFocusTarget = useCallback(() => {
    const selectedId = store.getState().selection[0];
    const node = selectedId
      ? canvasArea.current?.querySelector<HTMLElement>(
          `.react-flow__node[data-id="${CSS.escape(selectedId)}"]`,
        )
      : null;
    return node ?? region.current ?? false;
  }, [store]);
  const closeMenu = useCallback(
    (restoreFocus: boolean) => {
      setIntent(null);
      if (restoreFocus)
        requestAnimationFrame(() => {
          const target = canvasFocusTarget();
          if (target) target.focus({ preventScroll: true });
        });
    },
    [canvasFocusTarget],
  );
  const closeContext = useCallback(() => setContext(null), []);
  const transfer = useCanvasTransfer(store, !!editable, (screen) => {
    const rect = canvasArea.current!.getBoundingClientRect();
    return flow.screenToFlowPosition(
      screen ?? {
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
      },
    );
  });
  const perform = useCallback(
    (label: string, operations: CanvasOperation[]) => {
      if (!operations.length) return;
      void store
        .getState()
        .edit(label, operations)
        .catch((error) => toast.error(getApiErrorMessage(error)));
    },
    [store],
  );
  const {
    transient,
    measurements,
    onNodesChange,
    onEdgesChange,
    onNodeDragStart,
    onNodeDragStop,
  } = useCanvasFlowMotion(store, perform);
  const addAtCenter = useCallback(() => {
    const rect = canvasArea.current!.getBoundingClientRect();
    const screen = {
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    };
    const anchor = Array.from(region.current!.querySelectorAll<HTMLElement>("[data-canvas-add]"))
      .find((element) => element.getBoundingClientRect().width > 0);
    const anchorRect = anchor?.getBoundingClientRect();
    const center = flow.screenToFlowPosition(screen);
    setIntent({ anchor, screen: anchorRect ? { x: anchorRect.left, y: anchorRect.bottom + 8 } : screen,
      position: { x: center.x - 150, y: center.y - 100 }, automatic: true });
  }, [flow]);
  const addAfter = useCallback(
    (id: string, portId: string, element: HTMLElement) => {
      const current = store.getState().snapshot!.graph;
      const node = current.nodes.find((item) => item.id === id)!;
      const position = worldPosition(node, current.nodes);
      const rect = element.getBoundingClientRect();
      setIntent({
        screen: { x: rect.right + 8, y: rect.bottom },
        position: {
          x: position.x + (node.layout.width ?? 300) + 80,
          y: position.y,
        },
        source: { nodeId: id, portId, configVersion: node.configVersion },
        automatic: true,
      });
    },
    [store],
  );
  const closePanel = useCallback(() => {
    onPanelNodeChange(null);
    store.getState().select([]);
    region.current?.focus({ preventScroll: true });
  }, [onPanelNodeChange, store]);
  // 快捷面板里的“下一步”：在右侧就近的空位创建已连好线的下游节点，并直接打开它的面板。
  const createNext = useCallback(
    (id: string, portId: string, choice: CanvasNodeChoice) => {
      const current = store.getState().snapshot!.graph;
      const node = current.nodes.find((item) => item.id === id)!;
      const origin = worldPosition(node, current.nodes);
      const ports = Math.max(choice.capability.inputs.length, choice.capability.outputs.length);
      const position = nearbyNodePosition(
        current.nodes,
        { x: origin.x + (node.layout.width ?? 300) + 80, y: origin.y },
        { width: 300, height: Math.max(320, 128 + 32 * ports) },
        Object.fromEntries(flow.getNodes().map((item) => [item.id, item.measured])),
      );
      void createNode(
        {
          screen: { x: 0, y: 0 },
          position,
          source: { nodeId: id, portId, configVersion: node.configVersion },
        },
        choice,
      );
    },
    [store, createNode, flow],
  );
  const { nodes, edges, hidden } = useCanvasFlowProjection({
    snapshot,
    catalog,
    selection,
    edgeSelection,
    transient,
    measurements,
    editable: canEdit,
    pointerMode,
    onPreview,
    onRunNode,
    onShowNodeRun,
    addAfter,
    focusRequest,
    agentChangedIds,
  });
  const relations = useCanvasBusinessRelations(snapshot.graph, hidden);
  const flowEdges = useMemo(
    () => [...edges, ...relations.edges],
    [edges, relations.edges],
  );
  const automaticLayout = useCanvasAutoLayout({
    store, hidden, relations: relations.visibleRelations,
    arrangeUnavailable: relations.arrangeUnavailable,
  });
  const highlighted = useCanvasChainHighlight(nodes, flowEdges, selection);
  const {
    connect,
    onConnectEnd,
    onConnectStart,
    onReconnectStart,
    isValidConnection,
  } = useCanvasConnections({
    store,
    editable: !!editable,
    flow,
    perform,
    setIntent,
  });

  const { remove, copy, group, arrange, lock, onKeyDown } =
    useCanvasEditActions({
      store,
      snapshot,
      editable: !!editable,
      hidden,
      region,
      perform,
      onAdd: addAtCenter,
      onInspect: inspect,
      onMode: setPointerMode,
      onSearch: () => setSearching(true),
      onFit: fit,
      onHelp: () => setHelp(true),
      onAutoArrange: automaticLayout.arrange,
      onEscape: () => {
        setIntent(null);
        setContext(null);
      },
    });

  const annotate = () => {
    if (selection.length !== 2) return;
    const [source, target] = selection;
    const nodes = [source, target].map(
      (id) => snapshot.graph.nodes.find((node) => node.id === id)!,
    );
    perform("添加说明连线", [
      ...nodes.map((node) => ({
        type: "canvas.node.assert" as const,
        nodeId: node.id,
        expectedConfigVersion: node.configVersion,
      })),
      {
        type: "canvas.edge.create",
        edge: {
          id: crypto.randomUUID(),
          kind: "annotation",
          sourceNodeId: source,
          targetNodeId: target,
          sourcePort: null,
          targetPort: null,
          inputIndex: null,
          output: null,
          version: 0,
        },
      },
    ]);
  };
  const selectedNodes = snapshot.graph.nodes.filter((node) =>
    selection.includes(node.id),
  );
  const selectionActions = buildCanvasSelectionActions({
    nodes: selectedNodes,
    edgeCount: edgeSelection.length,
    editable: !!editable,
    compare: () => setComparing([...selection]),
    annotate,
    copy,
    lock,
    perform,
    remove,
    group,
    arrange,
    capability: catalog.nodes.find((item) => item.kind === selectedNodes[0]?.kind),
    addAfter,
  });
  const toolbarAnchors = selection.length
    ? selection
    : snapshot.graph.edges
        .filter((edge) => edgeSelection.includes(edge.id))
        .flatMap((edge) => [edge.sourceNodeId, edge.targetNodeId]);
  const onNodeClick = useCallback<
    NonNullable<ReactFlowProps<CanvasFlowNode>["onNodeClick"]>
  >(
    (event, node) => {
      if (
        pointerMode === "select" &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !isEditableTarget(event.target) &&
        !(
          event.target instanceof Element &&
          event.target.closest("button,a,.react-flow__handle")
        )
      )
        inspect(node.id);
    },
    [inspect, pointerMode],
  );
  const onEdgeClick = useCallback<
    NonNullable<ReactFlowProps<CanvasFlowNode>["onEdgeClick"]>
  >(
    (event, edge) => {
      if (
        pointerMode === "select" &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !edge.id.startsWith("relation:")
      )
        onInspectEdge(edge.id);
    },
    [onInspectEdge, pointerMode],
  );
  const onNodeContextMenu = useCallback<
    NonNullable<ReactFlowProps<CanvasFlowNode>["onNodeContextMenu"]>
  >(
    (event, node) => {
      if (isEditableTarget(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
      setIntent(null);
      if (!store.getState().selection.includes(node.id))
        store.getState().select([node.id]);
      setContext({
        position: { x: event.clientX, y: event.clientY },
        nodeId: node.id,
      });
    },
    [store],
  );
  const onEdgeContextMenu = useCallback<
    NonNullable<ReactFlowProps<CanvasFlowNode>["onEdgeContextMenu"]>
  >(
    (event, edge) => {
      if (edge.id.startsWith("relation:")) return;
      event.preventDefault();
      store.getState().select([], [edge.id]);
      setContext({
        position: { x: event.clientX, y: event.clientY },
        edgeId: edge.id,
      });
    },
    [store],
  );
  return (
    <div
      ref={region}
      className="flex min-h-0 grow basis-0 flex-col"
      tabIndex={0}
      role="region"
      aria-label="创作画布"
      onCopy={transfer.onCopy}
      onPaste={transfer.onPaste}
      onKeyDownCapture={(event) => {
        if (event.target instanceof Element && event.target.closest("[data-canvas-generation-composer]")) {
          return;
        }
        if (!gestureGate.keyDown(event)) onKeyDown(event);
      }}
    >
      <div
        ref={canvasArea}
        className="relative min-h-0 grow basis-0"
        {...edgeMarquee}
        onDragOver={transfer.onDragOver}
        onDrop={transfer.onDrop}
        onMouseDownCapture={gestureGate.mouseDown}
        onTouchStartCapture={gestureGate.touchStart}
        onClickCapture={gestureGate.click}
      >
        <div className="pointer-events-none absolute inset-x-3 top-3 z-20 flex flex-col items-start gap-2">
          <CanvasToolbar
            mode={pointerMode}
            onMode={setPointerMode}
            minimap={minimap}
            onMinimap={() => setMinimap(!minimap)}
            relations={relations.scope}
            onRelations={relations.setScope}
            relationAvailability={relations.available}
            relationLoading={relations.loading}
            relationError={relations.error}
            onReloadRelations={relations.refresh}
            onAdd={addAtCenter}
            onFit={() => fit()}
            onArrange={automaticLayout.arrange}
            arrangeError={automaticLayout.error}
            arrangeUnavailable={relations.arrangeUnavailable}
            onSearch={() => setSearching(true)}
          />
          {relations.scope !== "hidden" && (relations.error || relations.loading) && (
            <p
              role={relations.error ? "alert" : "status"}
              className="max-w-full rounded-2xl border border-border/40 bg-popover/90 px-3 py-2 text-xs text-muted-foreground shadow-sm backdrop-blur-xl"
            >
              {relations.error ?? "正在加载辅助连线…"}
            </p>
          )}
          <CanvasFlowNotices transfer={transfer} />
        </div>
        <ReactFlow<CanvasFlowNode>
          className={highlighted.active ? "fusion-canvas canvas-chain-active" : "fusion-canvas"}
          data-panning={pointerMode === "pan" || panKeyPressed || undefined}
          nodes={highlighted.nodes}
          edges={highlighted.edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnectStart={onConnectStart}
          connectionLineType={ConnectionLineType.Bezier}
          onConnect={(connection) => connect(connection)}
          onConnectEnd={onConnectEnd}
          onReconnectStart={onReconnectStart}
          onReconnect={(edge, connection) =>
            connect(
              connection,
              snapshot.graph.edges.find((item) => item.id === edge.id),
            )
          }
          isValidConnection={isValidConnection}
          onNodeDragStart={(event, node, nodes) => {
            onPanelNodeChange(null);
            onNodeDragStart(event, node, nodes);
          }}
          onNodeDragStop={onNodeDragStop}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneContextMenu={(event) => {
            if (!editable || isEditableTarget(event.target)) return;
            event.preventDefault();
            const screen = { x: event.clientX, y: event.clientY };
            setContext(null);
            setIntent({ screen, position: flow.screenToFlowPosition(screen) });
          }}
          onNodeContextMenu={onNodeContextMenu}
          onEdgeContextMenu={onEdgeContextMenu}
          onPaneClick={() => {
            onPanelNodeChange(null);
            setIntent(null);
            setContext(null);
          }}
          onMoveStart={() => setIntent(null)}
          onMoveEnd={(_event, viewport) => saveViewport(viewport)}
          onInit={() => {
            if (initialView.error) toast.error(initialView.error);
          }}
          defaultViewport={initialView.viewport}
          ariaLabelConfig={canvasAriaLabels}
          colorMode={resolvedTheme === "dark" ? "dark" : "light"}
          minZoom={CANVAS_MIN_ZOOM}
          maxZoom={CANVAS_MAX_ZOOM}
          panOnScroll={false}
          zoomOnScroll
          zoomOnPinch
          zoomOnDoubleClick={false}
          panOnDrag={pointerMode === "pan" ? [0, 1] : [1]}
          panActivationKeyCode="Space"
          selectionOnDrag={pointerMode === "select"}
          selectionMode={SelectionMode.Partial}
          multiSelectionKeyCode={["Control", "Meta", "Shift"]}
          deleteKeyCode={null}
          nodesDraggable={!!canEdit}
          nodesConnectable={!!canEdit}
          edgesReconnectable={!!canEdit}
          onlyRenderVisibleElements
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1.2} />
          {minimap && (
            <MiniMap<CanvasFlowNode>
              pannable
              zoomable
              position="bottom-right"
              ariaLabel="小地图，描边框表示当前视野，可拖动定位"
              nodeColor={(node) => canvasNodeColor(node.data.node.kind)}
              nodeStrokeColor={(node) =>
                node.selected
                  ? "var(--foreground)"
                  : canvasNodeColor(node.data.node.kind)
              }
              nodeStrokeWidth={2}
              maskColor="color-mix(in srgb, var(--background) 75%, transparent)"
              maskStrokeColor={
                resolvedTheme === "dark"
                  ? "var(--foreground)"
                  : "var(--primary)"
              }
              maskStrokeWidth={3}
            />
          )}
          {!context && (
            <CanvasSelectionActions
              nodeIds={toolbarAnchors}
              count={selection.length}
              edgeCount={edgeSelection.length}
              actions={selectionActions}
            />
          )}
          {panelNode && (
            <CanvasNodePanel node={panelNode} area={canvasArea} onClose={closePanel}>
              {(maxHeight) => {
                const panelKind = canvasNodePanelKind(panelNode);
                if (panelKind === "generation") {
                  return (
                    <GenerationComposer
                      key={panelNode.id}
                      node={panelNode}
                      maxHeight={maxHeight}
                      onClose={closePanel}
                      onPreview={onPreview}
                      onRun={onRunNode}
                    />
                  );
                }
                if (panelKind === "text") {
                  return (
                    <TextComposer
                      key={panelNode.id}
                      node={panelNode}
                      maxHeight={maxHeight}
                      onClose={closePanel}
                      onCreateNext={(portId, choice) =>
                        createNext(panelNode.id, portId, choice)
                      }
                    />
                  );
                }
                if (panelKind === "media") {
                  return (
                    <MediaNodePanel
                      key={panelNode.id}
                      node={panelNode}
                      maxHeight={maxHeight}
                      onClose={closePanel}
                      onPreview={onPreview}
                      onCreateNext={(portId, choice) =>
                        createNext(panelNode.id, portId, choice)
                      }
                    />
                  );
                }
                return (
                  <NodeFloatingPanel
                    key={panelNode.id}
                    node={panelNode}
                    maxHeight={maxHeight}
                    onClose={closePanel}
                    onPreview={onPreview}
                    onRun={onRunNode}
                  />
                );
              }}
            </CanvasNodePanel>
          )}
        </ReactFlow>
        {!snapshot.graph.nodes.length && (
          <CanvasEmptyState editable={!!editable} onAdd={addAtCenter} />
        )}
        <CanvasViewportStatus mode={pointerMode} onHelp={() => setHelp(true)} />
      </div>
      <CanvasAssistantBridge flow={flow} area={canvasArea} />
      {searching && (
        <CanvasSearchDialog
          onClose={() => setSearching(false)}
          onInspect={inspect}
        />
      )}
      {help && <CanvasHelpDialog onClose={() => setHelp(false)} />}
      {comparing && (
        <CanvasCompareDialog
          ids={comparing}
          onClose={() => setComparing(null)}
        />
      )}
      {intent && (
        <CanvasNodeMenu
          intent={intent}
          onClose={closeMenu}
          onChoose={(choice) => {
            setIntent(null);
            void createNode(intent, choice);
          }}
        />
      )}
      {context && (
        <CanvasContextActions
          position={context.position}
          actions={selectionActions}
          onClose={closeContext}
        />
      )}
    </div>
  );
}
