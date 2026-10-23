package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;

public sealed interface ResourceCommand permits ResourcePatchCommand, ResourceCreateCommand,
        ResourceDeleteCommand, InitializeProjectWorkspaceCommand, ScriptSourceCommand, ScriptScenesCommand, AssetBatchCreateCommand,
        StoryboardChildrenCommand, StoryboardBatchPatchCommand, ResourceBindMediaCommand, ComposeWritebackCommand, AdoptEpisodeMediaCommand, ResourceRestoreCommand, ProjectForkCommand, TemplateImportCommand {
    String requestId();

    CanvasValueType resourceType();

    ResourceOperation operation();
}
