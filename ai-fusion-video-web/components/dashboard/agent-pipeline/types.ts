import type { AiChatReq } from "@/lib/api/ai-pipeline";
import type {
  SubTimelineItem,
  TimelineItem,
} from "@/lib/store/pipeline-store";

export type { SubTimelineItem, TimelineItem };

export interface AgentPipelineState {
  status: "idle" | "reasoning" | "running" | "done" | "error" | "cancelled";
  reasoningText: string;
  reasoningDurationMs?: number;
  timeline: TimelineItem[];
  runId?: string;
  lastSequence: number;
  conversationId?: string;
  error?: string;
}

export interface AgentPipelineProps {
  request: AiChatReq;
  autoStart?: boolean;
  onComplete?: (conversationId?: string) => void;
  onError?: (error: string) => void;
}
