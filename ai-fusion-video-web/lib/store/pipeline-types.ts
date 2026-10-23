import type { AiChatReq, PipelineRunStatus } from "@/lib/api/ai-pipeline";
import type { TimelineItem } from "./pipeline-timeline";

export interface PipelineState {
  status: "running" | "done" | "error" | "cancelled";
  reasoningText: string;
  reasoningStartTime?: number;
  reasoningDurationMs?: number;
  timeline: TimelineItem[];
  runId?: string;
  lastSequence: number;
  conversationId?: string;
  error?: string;
  /** 页面恢复的 Pipeline 在打开详情前不回放 journal 内容。 */
  contentLoaded?: boolean;
  contentLoading?: boolean;
  /** 仅在详情选中时建立前端 reconnect SSE，不影响后端 Pipeline。 */
  reconnectOnSelect?: boolean;
  /** 页面列表/轮询得到的后端 run 状态。 */
  runStatus?: PipelineRunStatus;
  /** 取消请求已提交，等待 durable terminal event。 */
  cancelRequested?: boolean;
  /** 新一轮输出需要与上一轮末尾的文本分开渲染。 */
  separateNextRootContent?: boolean;
}

export interface PipelineTask {
  id: string;
  label: string;
  projectId: number | null;
  status: "running" | "done" | "error" | "cancelled";
  state: PipelineState;
  createdAt: number;
  cancellable?: boolean;
  /** 当前浏览器仍持有启动配置，可在失败后继续同一会话。 */
  continuable?: boolean;
  /** 任务结束时间（done/error/cancelled 时记录） */
  finishedAt?: number;
}

// ========== Store ==========

export interface PipelineStoreState {
  tasks: PipelineTask[];
  notificationOpen: boolean;
  /** 是否显示大面板（任务中心） */
  panelExpanded: boolean;
  /** 当前展开详情的 pipeline id */
  expandedTaskId: string | null;
  /** 是否已恢复过运行中 Pipeline 列表 */
  runningPipelinesRestored: boolean;

  // actions
  addPipeline: (config: {
    label: string;
    projectId: number;
    request: AiChatReq;
    onComplete?: () => void;
  }) => string;
  addPipelineContinuation: (config: {
    label: string;
    projectId: number | null;
    conversationId: string;
    initialTimeline?: TimelineItem[];
  }) => string;
  attachTaskStream: (config: {
    label: string;
    projectId: number;
    taskId: string;
    cancellable?: boolean;
    onComplete?: () => void;
    onSettled?: (status: "done" | "error" | "cancelled") => void;
  }) => string;
  /**
   * 添加一个非 AI 任务（如视频合成），不走 SSE 流。
   * 调用方负责后续通过 markSimpleTask 推进状态。
   */
  addSimpleTask: (config: {
    label: string;
    projectId: number;
    initialNote?: string;
    onComplete?: () => void;
  }) => string;
  /** 推进非 AI 任务的状态（追加结果文本/错误，标记完成或失败） */
  markSimpleTask: (
    id: string,
    update: {
      status: "done" | "error";
      resultText?: string;
      errorText?: string;
    }
  ) => void;
  cancelPipeline: (id: string) => Promise<void>;
  continuePipeline: (id: string) => void;
  removePipeline: (id: string) => void;
  clearCompleted: () => void;
  setNotificationOpen: (open: boolean) => void;
  setPanelExpanded: (expanded: boolean) => void;
  setExpandedTaskId: (id: string | null) => void;
  /** 首次打开恢复任务详情时，从 journal 起点加载完整内容。 */
  loadPipelineContent: (id: string) => void;
  /** 仅终止前端内容流；不会取消后端 Pipeline。 */
  disconnectPipelineContent: (id: string) => void;
  /** 页面加载时调用：查询 running 对话并轮询状态，不建立 SSE。 */
  restoreRunningPipelines: () => void;
  /** 页面重新可见时，从 durable cursor 恢复本地实时内容流。 */
  resumePipelineConnections: () => void;
}
