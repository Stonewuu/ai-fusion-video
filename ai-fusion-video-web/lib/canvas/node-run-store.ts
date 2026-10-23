import { createStore } from "zustand/vanilla";
import type { CanvasNodeActivity, CanvasNodeRun } from "@/lib/api/canvas-runs";

interface CanvasNodeRunState {
  ready: boolean;
  setReady: (ready: boolean) => void;
  nodes: ReadonlyMap<string, CanvasNodeRun>;
  children: ReadonlyMap<string, readonly CanvasNodeRun[]>;
  activities: ReadonlyMap<string, CanvasNodeActivity>;
  updateActivities: (activities: CanvasNodeActivity[]) => void;
  update: (nodes: CanvasNodeRun[]) => void;
}

/** Node bodies subscribe to their own versioned result, not the whole run panel. */
export function createCanvasNodeRunStore() {
  return createStore<CanvasNodeRunState>((set, get) => ({
    ready: false,
    setReady: (ready) => { if (ready !== get().ready) set({ ready }); },
    nodes: new Map(),
    children: new Map(),
    activities: new Map(),
    updateActivities: (activities) => {
      get().update(activities.flatMap((item) => [item.node, ...item.children]));
      const current = get();
      const next = new Map<string, CanvasNodeActivity>();
      for (const item of activities) {
        const node = current.nodes.get(item.node.nodeId)!;
        const old = current.activities.get(node.nodeId);
        const nested = current.children.get(node.id) ?? [];
        if (old?.runId === item.runId && old.runStatus === item.runStatus && old.canControl === item.canControl
          && old.node === node && old.children.length === nested.length && old.children.every((child, index) => child === nested[index])) next.set(node.nodeId, old);
        else next.set(node.nodeId, { ...item, node, children: [...nested] });
      }
      if (next.size !== current.activities.size || [...next].some(([id, item]) => current.activities.get(id) !== item)) set({ activities: next });
    },
    update: (nodes) => {
      const previous = get().nodes;
      const next = new Map<string, CanvasNodeRun>();
      const previousChildren = get().children;
      const childGroups = new Map<string, Map<number, CanvasNodeRun>>();
      for (const node of nodes) {
        if (node.parentNodeRunId !== null) {
          const group =
            childGroups.get(node.parentNodeRunId) ??
            new Map<number, CanvasNodeRun>();
          const prior = group.get(node.itemIndex);
          if (!prior || prior.attempt < node.attempt)
            group.set(node.itemIndex, node);
          childGroups.set(node.parentNodeRunId, group);
          continue;
        }
        if (next.has(node.nodeId)) continue;
        const old = previous.get(node.nodeId);
        // Derived availability is evaluated at read time, independently of the run revision.
        const unchanged =
          old?.id === node.id &&
          old.revision === node.revision &&
          old.status === node.status &&
          old.derivedState === node.derivedState &&
          old.contentUnavailable === node.contentUnavailable;
        next.set(node.nodeId, unchanged ? old : node);
      }
      const children = new Map<string, readonly CanvasNodeRun[]>();
      for (const [parentId, group] of childGroups) {
        const previousGroup = previousChildren.get(parentId);
        const oldById = new Map(previousGroup?.map((item) => [item.id, item]));
        const items = [...group.values()]
          .sort((a, b) => a.itemIndex - b.itemIndex)
          .map((item) => {
            const old = oldById.get(item.id);
            return old?.revision === item.revision &&
              old.status === item.status &&
              old.derivedState === item.derivedState &&
              old.contentUnavailable === item.contentUnavailable
              ? old
              : item;
          });
        children.set(
          parentId,
          previousGroup?.length === items.length &&
            items.every((item, index) => previousGroup[index] === item)
            ? previousGroup
            : items,
        );
      }
      if (
        previous.size === next.size &&
        [...next].every(([id, node]) => previous.get(id) === node) &&
        previousChildren.size === children.size &&
        [...children].every(([id, items]) => previousChildren.get(id) === items)
      )
        return;
      set({ nodes: next, children });
    },
  }));
}

export type CanvasNodeRunStore = ReturnType<typeof createCanvasNodeRunStore>;
