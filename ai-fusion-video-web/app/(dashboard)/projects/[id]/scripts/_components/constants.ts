import {
  MessageSquare,
  Eye,
  Volume2,
  Camera,
  TreePalm,
} from "lucide-react";

// ========== 对白类型配置 ==========

export const dialogueTypeConfig: Record<
  number,
  { icon: typeof MessageSquare; label: string; color: string; cls: string }
> = {
  1: { icon: MessageSquare, label: "对白", color: "blue", cls: "text-blue-400 border-blue-500/20 bg-blue-500/5" },
  2: { icon: Eye, label: "动作", color: "amber", cls: "text-amber-400 border-amber-500/20 bg-amber-500/5" },
  3: { icon: Volume2, label: "旁白", color: "purple", cls: "text-purple-400 border-purple-500/20 bg-purple-500/5" },
  4: { icon: Camera, label: "镜头", color: "cyan", cls: "text-cyan-400 border-cyan-500/20 bg-cyan-500/5" },
  5: { icon: TreePalm, label: "环境", color: "green", cls: "text-green-400 border-green-500/20 bg-green-500/5" },
};
