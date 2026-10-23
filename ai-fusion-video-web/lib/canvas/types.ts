import type { ResourceType } from "@/lib/api/resource-events";
import type { AgentSkillOption } from "@/lib/api/agent-definitions";

export type CanvasValueType =
  | ResourceType
  | "text"
  | "image"
  | "video"
  | "audio"
  | "document";
// 文档保持独立类型；只有文本文件输出 text，PDF 不隐式转换为提示词。
export type CanvasNodeKind =
  | CanvasValueType
  | "story"
  | "textFile"
  | "textGeneration"
  | "textCombine"
  | "scriptParse"
  | "storyToScript"
  | "episodeParse"
  | "storyboardGenerate"
  | "assetExtract"
  | "assetImageGenerate"
  | "frameGenerate"
  | "imageGeneration"
  | "videoGeneration"
  | "comfyui"
  | "frameExtract"
  | "imageTransform"
  | "musicGeneration"
  | "sfxGeneration"
  | "audioMix"
  | "videoCompose"
  | "export"
  | "agent"
  | "mcpTool"
  | "selection"
  | "collection"
  | "foreach"
  | "writeback"
  | "group"
  | "note";
export type CanvasConfig = {
  [key: string]: string | number | boolean | string[] | CanvasConfig;
};
export interface CanvasResourceRef {
  type: ResourceType;
  id: number;
  version: number;
}
export interface CanvasLayout {
  x: number;
  y: number;
  width: number | null;
  height: number | null;
  parentId: string | null;
  locked: boolean;
  collapsed: boolean;
}
export interface CanvasNode {
  id: string;
  kind: CanvasNodeKind;
  schemaVersion: number;
  title: string;
  layout: CanvasLayout;
  config: CanvasConfig;
  resource: CanvasResourceRef | null;
  configVersion: number;
  layoutVersion: number;
}
export interface CanvasEdge {
  id: string;
  kind: "execution" | "reference" | "annotation";
  sourceNodeId: string;
  sourcePort: string | null;
  targetNodeId: string;
  targetPort: string | null;
  inputIndex: number | null;
  output: { nodeRunId: string | null; outputId: string; resourceId: string | null; snapshotId?: string | null } | null;
  version: number;
}
export interface CanvasGraph {
  schemaVersion: number;
  projectId: number | null;
  nodes: CanvasNode[];
  edges: CanvasEdge[];
}
export interface CanvasDocument {
  id: string;
  title: string;
  projectId: number | null;
  ownerType: number;
  ownerId: number;
  createdBy: number;
  schemaVersion: number;
  revision: number;
  nextEventSequence: number;
  archived: boolean;
  createTime: string;
  updateTime: string;
}
export interface CanvasSnapshot {
  document: CanvasDocument;
  graph: CanvasGraph;
  lastSequence: number;
  permissions: {
    canView: boolean;
    canEdit: boolean;
    canExecute: boolean;
    canManage: boolean;
  };
}
export interface CanvasConfigField {
  type: "string" | "integer" | "number" | "boolean" | "object" | "string_array";
  required: boolean;
  minimum: number | null;
  maximum: number | null;
  maxLength: number | null;
  enumValues: string[];
}
export interface CanvasPort {
  id: string;
  label: string;
  valueType: CanvasValueType | null;
  valueTypeField: string | null;
  acceptedTypes: CanvasValueType[];
  collection: boolean;
  required: boolean;
  maxConnections: number;
}
export interface CanvasCapability {
  kind: CanvasNodeKind;
  label: string;
  category: string;
  executor: string;
  schemaVersion: number;
  requiresProject: boolean;
  resourceType: ResourceType | null;
  configFields: Record<string, CanvasConfigField>;
  inputs: CanvasPort[];
  outputs: CanvasPort[];
}
export interface CanvasCatalog {
  schemaVersion: number;
  nodes: CanvasCapability[];
  models: CanvasModelOption[];
  audioServices: { id: number; name: string }[];
  writebackFields: CanvasWritebackField[];
  agentProfiles: CanvasAgentProfile[];
  skills: AgentSkillOption[];
}
export interface CanvasAgentProfile {
  type: string;
  name: string;
  customized: boolean;
  toolsEnabled: boolean;
  defaultSkills: string[];
  generationKinds: ("image" | "video")[];
  inputTypes: CanvasValueType[];
  singleInput: boolean;
  requiresTypedInput: boolean;
  resultTypes: CanvasValueType[];
  hint: string;
}
export interface CanvasWritebackField {
  key: string;
  label: string;
  targetType: ResourceType;
  valueType: "text" | "image" | "video" | "audio";
  path: string;
}
export interface CanvasModelInput {
  maximum: number | null;
  minimum: number;
  minimumWhenProvided?: number;
}
export interface CanvasModelVersion {
  id: number;
  number: number;
  inputs: Record<string, CanvasModelInput>;
  fields: string[];
  parameters: Record<string, unknown>;
  unavailable: string | null;
}
export interface CanvasModelOption {
  id: number;
  name: string;
  kind: string;
  resultType: string;
  inputs: Record<string, CanvasModelInput>;
  parameters: Record<string, unknown>;
  versions: CanvasModelVersion[];
  unavailable: string | null;
}
export type CanvasOperation =
  | {
      type: "canvas.node.assert";
      nodeId: string;
      expectedConfigVersion: number;
    }
  | { type: "canvas.node.create"; node: CanvasNode }
  | {
      type: "canvas.node.patch";
      nodeId: string;
      expectedConfigVersion: number;
      title?: string;
      configPatch?: Record<string, CanvasConfig[string] | null>;
    }
  | {
      type: "canvas.node.layout";
      nodeId: string;
      expectedLayoutVersion: number;
      layout: CanvasLayout;
    }
  | {
      type: "canvas.node.bind";
      nodeId: string;
      expectedConfigVersion: number;
      resource: CanvasResourceRef;
    }
  | {
      type: "canvas.node.delete";
      nodeId: string;
      expectedConfigVersion: number;
      expectedLayoutVersion: number;
    }
  | { type: "canvas.edge.create"; edge: CanvasEdge }
  | { type: "canvas.edge.replace"; edge: CanvasEdge; expectedVersion: number }
  | { type: "canvas.edge.delete"; edgeId: string; expectedVersion: number };
export interface CanvasEdit {
  requestId: string;
  operations: CanvasOperation[];
}
export interface CanvasChanges {
  upsertedNodes: CanvasNode[];
  deletedNodeIds: string[];
  upsertedEdges: CanvasEdge[];
  deletedEdgeIds: string[];
}
export interface CanvasReceipt {
  commandId: string;
  requestId: string;
  canvasId: string;
  revision: number;
  sequence: number;
  changes: CanvasChanges;
}
export interface CanvasEvent {
  id: string;
  canvasId: string;
  sequence: number;
  commandId: string | null;
  type: string;
  payload: Record<string, unknown>;
  createdAt: string;
}
export interface CanvasEventPage {
  events: CanvasEvent[];
  latestSequence: number;
  hasMore: boolean;
}
export interface CanvasResource {
  header: {
    resourceType: ResourceType;
    resourceId: number;
    projectId: number | null;
    version: number;
    deleted: boolean;
  };
  data: Record<string, unknown>;
  canEdit?: boolean;
}
export interface CanvasHistory {
  commandId: string;
  revision: number;
  actorUserId: number;
  actorName: string;
  origin: string;
  eventType: string;
  nodeIds: string[];
  edgeCount: number;
  canUndo: boolean;
  createdAt: string;
}
export interface CanvasComment {
  id: string;
  nodeId: string | null;
  parentId: string | null;
  resourceVersion: number | null;
  content: string;
  createdBy: number;
  authorName: string;
  resolved: boolean;
  version: number;
  deleted: boolean;
  createdAt: string;
  updatedAt: string;
}
export interface CanvasCheckpoint {
  id: string;
  title: string;
  revision: number;
  sequence: number;
  createdBy: number;
  authorName: string;
  createdAt: string;
}
export interface CanvasCheckpointDetail {
  checkpoint: CanvasCheckpoint;
  snapshot: { graph: CanvasGraph; resources: CanvasResource[] };
}

export interface CanvasSourceIntent {
  nodeId: string;
  portId: string;
  configVersion: number;
}
export interface CanvasAddIntent {
  anchor?: HTMLElement;
  screen: { x: number; y: number };
  position: { x: number; y: number };
  source?: CanvasSourceIntent;
  automatic?: boolean;
}
