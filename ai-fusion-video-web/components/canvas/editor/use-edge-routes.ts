"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useReactFlow, type Edge } from "@xyflow/react";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { layoutRequestKey } from "@/lib/canvas/layout/request-key";
import { runLayoutWorker } from "@/lib/canvas/layout/worker-client";
import type { LayoutRelation, LayoutRequest, LayoutResult } from "@/lib/canvas/layout/types";
import type { CanvasCatalog, CanvasGraph } from "@/lib/canvas/types";
import { captureLayoutGeometry, captureLayoutLabels } from "./layout-geometry";
import type { Measurements } from "./use-flow-motion";

export function useCanvasEdgeRoutes({ graph, catalog, relations, hidden, edges, measurements, paused }: {
  graph: CanvasGraph;
  catalog: CanvasCatalog;
  relations: LayoutRelation[];
  hidden: Set<string>;
  edges: Edge[];
  measurements: Measurements;
  paused: boolean;
}) {
  const flow = useReactFlow();
  const [routes, setRoutes] = useState<LayoutResult["routes"]>({});
  const [error, setError] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(0);
  const completed = useRef<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const accept = useCallback((request: LayoutRequest, result: LayoutResult) => {
    controller.current?.abort();
    completed.current = layoutRequestKey(request);
    setRoutes(result.routes);
    setError(null);
    setAccepted((value) => value + 1);
  }, []);
  useEffect(() => {
    if (paused || !graph.nodes.length) {
      return;
    }
    const abort = new AbortController();
    controller.current = abort;
    // 合并连续测量和编辑；缩放、平移、拖动期间不重新寻找路径。
    const timer = setTimeout(async () => {
      try {
        const request: LayoutRequest = {
          graph, geometry: captureLayoutGeometry(graph, catalog, flow),
          selection: [], relations, hidden: [...hidden],
          labels: captureLayoutLabels(flow),
        };
        const key = layoutRequestKey(request);
        if (completed.current === key) {
          return;
        }
        const result = await runLayoutWorker(request, "route", abort.signal);
        if (!abort.signal.aborted) {
          completed.current = key;
          setRoutes(result.routes);
          setError(null);
        }
      } catch (failure) {
        if (!abort.signal.aborted) {
          setError(getApiErrorMessage(failure));
        }
      }
    }, 160);
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [graph, catalog, relations, hidden, measurements, paused, flow, accepted]);
  const routedEdges = useMemo(() => edges.map((edge) => ({
    ...edge, type: "canvas", data: { ...edge.data, route: routes[edge.id] },
  })), [edges, routes]);
  return { edges: routedEdges, error: edges.length ? error : null, accept };
}
