"use client";
import { useCallback, useRef, type Dispatch, type SetStateAction } from "react";
import {
  type Connection,
  type OnConnectEnd,
  type OnConnectStart,
  type ReactFlowProps,
  type IsValidConnection,
  type ReactFlowInstance,
} from "@xyflow/react";
import { toast } from "sonner";
import { validateConnection } from "@/lib/canvas/graph";
import type {
  CanvasAddIntent,
  CanvasEdge,
  CanvasOperation,
  CanvasSourceIntent,
} from "@/lib/canvas/types";
import type { CanvasStore } from "@/lib/store/canvas-store";
import type { CanvasFlowNode } from "./canvas-node";
export function useCanvasConnections({
  store,
  editable,
  flow,
  perform,
  setIntent,
}: {
  store: CanvasStore;
  editable: boolean;
  flow: ReactFlowInstance<CanvasFlowNode>;
  perform: (label: string, operations: CanvasOperation[]) => void;
  setIntent: Dispatch<SetStateAction<CanvasAddIntent | null>>;
}) {
  const connectionStart = useRef<
    (CanvasSourceIntent & { handleType: "source" | "target" }) | null
  >(null);
  const startVersions = useRef(new Map<string, number>());
  const reconnecting = useRef<CanvasEdge | null>(null);
  const captureVersions = () => {
    startVersions.current = new Map(
      store
        .getState()
        .snapshot!.graph.nodes.map((node) => [node.id, node.configVersion]),
    );
  };
  const connect = useCallback(
    (connection: Connection, previous?: CanvasEdge) => {
      const { source, target, sourceHandle, targetHandle } = connection;
      const state = store.getState();
      const graph = state.snapshot!.graph;
      if (!sourceHandle || !targetHandle) {
        toast.error("请选择具体输入和输出端口");
        return;
      }
      const checked = previous
        ? {
          ...graph,
          edges: graph.edges.filter((edge) => edge.id !== previous.id),
        }
        : graph;
      const error =
        validateConnection(
          checked,
          state.catalog!.nodes,
          source,
          sourceHandle,
          target,
          targetHandle,
        );
      if (error) {
        toast.error(error);
        return;
      }
      let inputIndex = 0;
      const used = new Set(
        checked.edges
          .filter(
            (edge) =>
              edge.kind === "execution" &&
              edge.targetNodeId === target &&
              edge.targetPort === targetHandle,
          )
          .map((edge) => edge.inputIndex),
      );
      while (used.has(inputIndex)) inputIndex++;
      if (
        previous &&
        previous.targetNodeId === target &&
        previous.targetPort === targetHandle &&
        previous.inputIndex !== null &&
        !used.has(previous.inputIndex)
      )
        inputIndex = previous.inputIndex;
      const edge: CanvasEdge = {
        id: previous?.id ?? crypto.randomUUID(),
        kind: "execution",
        sourceNodeId: source,
        targetNodeId: target,
        sourcePort: sourceHandle,
        targetPort: targetHandle,
        inputIndex,
        output: null,
        version: previous?.version ?? 0,
      };
      const sourceVersion =
        startVersions.current.get(source) ??
        graph.nodes.find((node) => node.id === source)!.configVersion;
      perform(previous ? "调整连线" : "连接节点", [
        {
          type: "canvas.node.assert",
          nodeId: source,
          expectedConfigVersion: sourceVersion,
        },
        {
          type: "canvas.node.assert",
          nodeId: target,
          expectedConfigVersion:
            startVersions.current.get(target) ??
            graph.nodes.find((node) => node.id === target)!.configVersion,
        },
        previous
          ? {
            type: "canvas.edge.replace",
            edge,
            expectedVersion:
              reconnecting.current?.id === previous.id
                ? reconnecting.current.version
                : previous.version,
          }
          : { type: "canvas.edge.create", edge },
      ]);
    },
    [perform, store],
  );
  const onConnectEnd: OnConnectEnd = useCallback(
    (event, state) => {
      const handleType = state.fromHandle?.type;
      if (
        state.isValid ||
        (handleType !== "source" && handleType !== "target") ||
        !state.fromNode ||
        !state.fromHandle?.id ||
        !editable
      ) {
        return;
      }
      if (
        !(event.target instanceof Element) ||
        !event.target.closest(".react-flow__pane")
      ) {
        return;
      }
      const pointer =
        "changedTouches" in event ? event.changedTouches[0] : event;
      const screen = { x: pointer.clientX, y: pointer.clientY };
      const origin = connectionStart.current;
      if (
        !origin ||
        origin.handleType !== handleType ||
        origin.nodeId !== state.fromNode.id ||
        origin.portId !== state.fromHandle.id
      ) {
        return;
      }
      const portIntent: CanvasSourceIntent = {
        nodeId: origin.nodeId,
        portId: origin.portId,
        configVersion: origin.configVersion,
      };
      if (handleType === "target") {
        const snapshot = store.getState().snapshot;
        const catalog = store.getState().catalog;
        const targetNode = snapshot?.graph.nodes.find(
          (node) => node.id === origin.nodeId,
        );
        const inputPort = catalog?.nodes
          .find((item) => item.kind === targetNode?.kind)
          ?.inputs.find((port) => port.id === origin.portId);
        if (!targetNode || !inputPort) {
          return;
        }
        if (Object.hasOwn(targetNode.config, inputPort.id)) {
          toast.error("该输入已有手填值，请先移除手填值后再连接前置节点");
          return;
        }
        const existing = snapshot!.graph.edges.filter(
          (edge) =>
            edge.kind === "execution" &&
            edge.targetNodeId === targetNode.id &&
            edge.targetPort === inputPort.id,
        ).length;
        if (existing >= inputPort.maxConnections) {
          toast.error("该输入已达到连接数量上限");
          return;
        }
        setIntent({
          screen,
          position: flow.screenToFlowPosition(screen),
          target: portIntent,
        });
        return;
      }
      setIntent({
        screen,
        position: flow.screenToFlowPosition(screen),
        source: portIntent,
      });
    },
    [editable, flow, setIntent, store],
  );
  const onConnectStart: OnConnectStart = (_event, params) => {
    captureVersions();
    reconnecting.current = null;
    const node = store
      .getState()
      .snapshot!.graph.nodes.find((item) => item.id === params.nodeId);
    connectionStart.current =
      node &&
        params.handleId &&
        (params.handleType === "source" || params.handleType === "target")
        ? {
          nodeId: node.id,
          portId: params.handleId,
          configVersion: node.configVersion,
          handleType: params.handleType,
        }
        : null;
  };
  const onReconnectStart: NonNullable<ReactFlowProps["onReconnectStart"]> = (
    _event,
    edge,
  ) => {
    captureVersions();
    reconnecting.current =
      store
        .getState()
        .snapshot!.graph.edges.find((item) => item.id === edge.id) ?? null;
    connectionStart.current = null;
  };
  const isValidConnection: IsValidConnection = (connection) => {
    const state = store.getState();
    const graph = state.snapshot!.graph;
    const checked = {
      ...graph,
      edges: graph.edges.filter((edge) => edge.id !== reconnecting.current?.id),
    };
    if (!connection.sourceHandle || !connection.targetHandle) return false;
    return (
      !validateConnection(
        checked,
        state.catalog!.nodes,
        connection.source,
        connection.sourceHandle,
        connection.target,
        connection.targetHandle,
      )
    );
  };
  return {
    connect,
    onConnectEnd,
    onConnectStart,
    onReconnectStart,
    isValidConnection,
  };
}
