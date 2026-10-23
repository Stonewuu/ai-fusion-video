import { http } from "./client";
import type { ResourceType } from "./resource-events";
import type {
  CanvasCatalog,
  CanvasCheckpoint,
  CanvasCheckpointDetail,
  CanvasComment,
  CanvasDocument,
  CanvasEdit,
  CanvasEventPage,
  CanvasHistory,
  CanvasReceipt,
  CanvasResource,
  CanvasSnapshot,
} from "@/lib/canvas/types";

const path = (id: string) => `/api/canvases/${id}`;
export const canvasApi = {
  capabilities: (): Promise<CanvasCatalog> =>
    http.get("/api/canvases/capabilities"),
  list: (
    params: {
      projectId?: number;
      archived?: boolean;
      pageNo?: number;
      pageSize?: number;
      search?: string;
    },
    signal?: AbortSignal,
  ): Promise<{ list: (CanvasDocument & { projectName: string | null })[]; total: number }> =>
    http.get("/api/canvases", { params, signal }),
  create: (
    id: string,
    title: string,
    projectId: number | null,
  ): Promise<CanvasDocument> =>
    http.post("/api/canvases", { id, title, projectId }),
  snapshot: (id: string, signal?: AbortSignal): Promise<CanvasSnapshot> =>
    http.get(path(id), { signal }),
  edit: (id: string, command: CanvasEdit): Promise<CanvasReceipt> =>
    http.post(`${path(id)}/commands`, command),
  undo: (
    id: string,
    commandId: string,
    requestId: string,
  ): Promise<CanvasReceipt> =>
    http.post(`${path(id)}/undo`, { commandId, requestId }),
  events: (
    id: string,
    afterSequence: number,
    signal?: AbortSignal,
  ): Promise<CanvasEventPage> =>
    http.get(`${path(id)}/events`, {
      params: { afterSequence, limit: 200 },
      signal,
    }),
  metadata: (
    id: string,
    command: {
      requestId: string;
      expectedRevision: number;
      title?: string;
      archived?: boolean;
    },
  ): Promise<CanvasReceipt> => http.put(`${path(id)}/metadata`, command),
  resource: (
    id: string,
    type: ResourceType,
    resourceId: number,
    signal?: AbortSignal,
  ): Promise<CanvasResource> =>
    http.get(`${path(id)}/resource-commands/${type}/${resourceId}`, { signal }),
  collection: (
    id: string,
    type: ResourceType,
    resourceId: number,
    depth = 1,
    signal?: AbortSignal,
  ): Promise<CanvasResourceCollection> =>
    http.get(`${path(id)}/collections/${type}/${resourceId}`, {
      params: { depth },
      signal,
    }),
  patchResource: (
    id: string,
    command: Record<string, unknown>,
  ): Promise<ResourceReceipt> =>
    http.post(`${path(id)}/resource-commands`, command),
  resourceCommand: (
    id: string,
    operation:
      | "create"
      | "delete"
      | "script-source"
      | "script-scenes"
      | "asset-batch"
      | "storyboard-children"
      | "storyboard-batch",
    command: Record<string, unknown>,
  ): Promise<ResourceReceipt> =>
    http.post(`${path(id)}/resource-commands/${operation}`, command),
  createAndPlace: (
    id: string,
    command: CanvasPlacementCommand,
  ): Promise<{ resource: ResourceReceipt; canvas: CanvasReceipt }> =>
    http.post(`${path(id)}/resource-commands/create-and-place`, command),
  history: (id: string, beforeRevision?: number): Promise<CanvasHistory[]> =>
    http.get(`${path(id)}/history`, { params: { beforeRevision, limit: 40 } }),
  comments: (id: string, offset = 0): Promise<CanvasComment[]> =>
    http.get(`${path(id)}/comments`, { params: { offset, limit: 50 } }),
  commentDetail: (id: string, commentId: string): Promise<CanvasComment> =>
    http.get(`${path(id)}/comments/${commentId}`),
  comment: (
    id: string,
    command: {
      requestId: string;
      id: string;
      nodeId: string | null;
      parentId: string | null;
      content: string;
    },
  ): Promise<CanvasReceipt> => http.post(`${path(id)}/comments`, command),
  patchComment: (
    id: string,
    commentId: string,
    command: {
      requestId: string;
      expectedVersion: number;
      content?: string;
      resolved?: boolean;
      deleted?: boolean;
    },
  ): Promise<CanvasReceipt> =>
    http.put(`${path(id)}/comments/${commentId}`, command),
  checkpoints: (
    id: string,
    beforeRevision?: number,
  ): Promise<CanvasCheckpoint[]> =>
    http.get(`${path(id)}/checkpoints`, {
      params: { beforeRevision, limit: 40 },
    }),
  checkpoint: (
    id: string,
    checkpointId: string,
  ): Promise<CanvasCheckpointDetail> =>
    http.get(`${path(id)}/checkpoints/${checkpointId}`),
  saveCheckpoint: (
    id: string,
    command: {
      requestId: string;
      id: string;
      title: string;
      expectedRevision: number;
    },
  ): Promise<CanvasReceipt> => http.post(`${path(id)}/checkpoints`, command),
};

export interface CanvasPlacementCommand {
  requestId: string;
  nodeId: string;
  title: string;
  resourceType: ResourceType;
  fields: Record<string, unknown>;
  x: number;
  y: number;
}

export interface CanvasCollectionEntry extends CanvasResource {
  parentType: ResourceType | null;
  parentId: number | null;
}
export interface CanvasResourceCollection {
  root: CanvasCollectionEntry;
  descendants: CanvasCollectionEntry[];
}

export interface ResourceReceipt {
  commandId: string;
  requestId: string;
  result: Record<string, unknown>;
  changes: {
    resourceType: ResourceType;
    resourceId: number;
    version: number;
    deleted: boolean;
  }[];
}
