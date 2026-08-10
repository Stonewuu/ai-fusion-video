"use client";

import { XCircle } from "lucide-react";
import { GenerationModelCapabilitiesResult } from "@/components/dashboard/generation-model-capabilities-result";
import {
  ImageGenerateResult,
  VideoGenerateResult,
} from "@/components/dashboard/generation-media-result";
import {
  AssetItemsResult,
  AssetListResult,
  BatchCreateResult,
  CreateResult,
  MetadataResult,
  MutationResult,
  ProjectInfoResult,
  SaveEpisodeResult,
} from "./resources";
import {
  EpisodeDetailResult,
  SceneDetailResult,
  ScriptInfoResult,
  ScriptStructureResult,
} from "./scripts";
import { GenericResult } from "./shared";
import { StoryboardResult } from "./storyboard";

export function ToolResultDisplay({
  toolName,
  content,
  expanded = false,
}: {
  toolName: string;
  content: string;
  expanded?: boolean;
}) {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    const isLong = content.length > 300;
    return (
      <div className="text-xs text-foreground/70 whitespace-pre-wrap leading-relaxed">
        {isLong ? `${content.slice(0, 300)}…` : content}
      </div>
    );
  }

  if (
    typeof parsed === "object" &&
    parsed !== null &&
    (parsed as Record<string, unknown>).status === "error"
  ) {
    const message = (parsed as Record<string, unknown>).message;
    return (
      <p className="text-xs text-destructive inline-flex items-center gap-1">
        <XCircle className="h-3.5 w-3.5 shrink-0" />
        {typeof message === "string" ? message : "操作失败"}
      </p>
    );
  }

  if (parsed === null) {
    return (
      <p className="text-xs text-muted-foreground">
        工具未返回结果
      </p>
    );
  }

  switch (toolName) {
    case "get_generation_model_capabilities":
      return <GenerationModelCapabilitiesResult data={parsed} />;
    case "generate_image":
      return <ImageGenerateResult data={parsed} />;
    case "generate_video":
      return <VideoGenerateResult data={parsed} />;
    case "list_project_assets":
      return <AssetListResult data={parsed} />;
    case "query_asset_metadata":
      return <MetadataResult data={parsed} />;
    case "batch_create_assets":
      return <BatchCreateResult data={parsed} />;
    case "get_project_script":
      return <ScriptInfoResult data={parsed} />;
    case "get_script_structure":
      return <ScriptStructureResult data={parsed} />;
    case "get_script_episode":
      return <EpisodeDetailResult data={parsed} />;
    case "get_script_scene":
      return <SceneDetailResult data={parsed} />;
    case "query_asset_items":
      return <AssetItemsResult data={parsed} />;
    case "save_script_episode":
      return <SaveEpisodeResult data={parsed} />;
    case "save_script_scene_items":
    case "save_project":
    case "delete_project":
    case "update_script_info":
    case "update_script_scene":
    case "manage_script_scenes":
    case "update_script":
    case "update_asset":
    case "update_asset_image":
    case "update_asset_item":
    case "delete_asset_resource":
    case "delete_script_child_resource":
    case "update_storyboard_scene":
    case "update_storyboard_item":
    case "update_storyboard_item_frame":
    case "update_storyboard_item_video":
    case "delete_storyboard_child_resource":
      return <MutationResult data={parsed} toolName={toolName} />;
    case "create_asset":
    case "add_asset_item":
    case "batch_create_asset_items":
      return <CreateResult data={parsed} toolName={toolName} />;
    case "get_project":
      return <ProjectInfoResult data={parsed} />;
    case "insert_storyboard_item":
    case "save_storyboard_episode":
    case "save_storyboard_scene_shots":
    case "get_storyboard":
    case "get_storyboard_scene_items":
    case "get_project_storyboard":
      return (
        <StoryboardResult
          data={parsed}
          toolName={toolName}
          expanded={expanded}
        />
      );
    case "get_script":
    case "get_asset":
    case "list_my_projects":
    default:
      return <GenericResult data={parsed} />;
  }
}
