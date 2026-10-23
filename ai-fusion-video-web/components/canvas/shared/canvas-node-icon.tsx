import { AudioLines, Bot, Box, Cable, FileText, ImageIcon, Layers, StickyNote, Video, type LucideProps } from "lucide-react";
import type { CanvasCapability } from "@/lib/canvas/types";

export function CanvasNodeTypeIcon({ capability, ...props }: LucideProps & { capability: CanvasCapability }) {
  if (capability.kind === "group") return <Layers {...props} />;
  if (capability.kind === "note") return <StickyNote {...props} />;
  if (capability.kind === "image" || capability.executor === "image") return <ImageIcon {...props} />;
  if (capability.kind === "video" || capability.executor === "video") return <Video {...props} />;
  if (capability.kind === "audio" || capability.executor === "audio") return <AudioLines {...props} />;
  if (capability.executor === "mcp") return <Cable {...props} />;
  if (capability.executor === "agent") return <Bot {...props} />;
  if (capability.resourceType) return <Box {...props} />;
  return <FileText {...props} />;
}
