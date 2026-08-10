import type { AiChatReq, PendingToolCallInfo } from "@/lib/api/ai-pipeline";
import type {
  SubTimelineItem,
  TimelineItem,
} from "@/lib/store/pipeline-store";

export type { SubTimelineItem, TimelineItem };

export interface AgentPipelineState {
  status: "idle" | "reasoning" | "running" | "cancelling" | "done" | "error" | "cancelled";
  reasoningText: string;
  reasoningStartTime?: number;
  reasoningDurationMs?: number;
  timeline: TimelineItem[];
  runId?: string;
  lastSequence: number;
  conversationId?: string;
  error?: string;
  pendingConfirmation?: {
    runId: string;
    replyId: string;
    parentToolCallId?: string;
    toolCalls: PendingToolCallInfo[];
    expiresAt: string;
    decisions: Record<string, boolean>;
    submitting: boolean;
  };
}

export interface AgentPipelineProps {
  request: AiChatReq;
  autoStart?: boolean;
  onComplete?: (conversationId?: string) => void;
  onError?: (error: string) => void;
}
