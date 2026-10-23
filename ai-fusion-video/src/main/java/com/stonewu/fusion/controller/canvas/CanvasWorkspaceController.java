package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository.HistoryEntry;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.Checkpoint;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.CheckpointDetail;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository.Comment;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasUndoService;
import com.stonewu.fusion.service.canvas.CanvasWorkspaceService;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCheckpointForkService;
import com.stonewu.fusion.service.canvas.collaboration.CanvasCheckpointService;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasUndoCommand;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CheckpointCreate;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CommentCreate;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CommentPatch;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.DocumentPatch;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/canvases/{canvasId}")
@RequiredArgsConstructor
public class CanvasWorkspaceController {
    private final CanvasActorResolver actors;
    private final CanvasWorkspaceService workspace;
    private final CanvasUndoService undo;
    private final CanvasCheckpointService checkpoints;
    private final CanvasCheckpointForkService checkpointForks;

    @PutMapping("/metadata")
    public CommonResult<CanvasCommandReceipt> metadata(@PathVariable String canvasId,
                                                       @Valid @RequestBody DocumentPatch command) {
        return CommonResult.success(workspace.patch(canvasId, command, actors.current()));
    }

    @PostMapping("/undo")
    public CommonResult<CanvasCommandReceipt> undo(@PathVariable String canvasId,
                                                   @Valid @RequestBody CanvasUndoCommand command) {
        return CommonResult.success(undo.undo(canvasId, command, actors.current()));
    }

    @GetMapping("/history")
    public CommonResult<List<HistoryEntry>> history(@PathVariable String canvasId,
                                                    @RequestParam(defaultValue = "9223372036854775807") long beforeRevision,
                                                    @RequestParam(defaultValue = "40") int limit) {
        return CommonResult.success(workspace.history(canvasId, actors.current(), beforeRevision, limit));
    }

    @GetMapping("/comments")
    public CommonResult<List<Comment>> comments(@PathVariable String canvasId,
                                                @RequestParam(defaultValue = "0") int offset,
                                                @RequestParam(defaultValue = "50") int limit) {
        return CommonResult.success(workspace.comments(canvasId, actors.current(), offset, limit));
    }

    @PostMapping("/comments")
    public CommonResult<CanvasCommandReceipt> comment(@PathVariable String canvasId,
                                                      @Valid @RequestBody CommentCreate command) {
        return CommonResult.success(workspace.comment(canvasId, command, actors.current()));
    }

    @GetMapping("/comments/{commentId}")
    public CommonResult<Comment> comment(@PathVariable String canvasId, @PathVariable String commentId) {
        return CommonResult.success(workspace.comment(canvasId, commentId, actors.current()));
    }

    @PutMapping("/comments/{commentId}")
    public CommonResult<CanvasCommandReceipt> comment(@PathVariable String canvasId, @PathVariable String commentId,
                                                      @Valid @RequestBody CommentPatch command) {
        return CommonResult.success(workspace.patchComment(canvasId, commentId, command, actors.current()));
    }

    @GetMapping("/checkpoints")
    public CommonResult<List<Checkpoint>> checkpoints(@PathVariable String canvasId,
                                                      @RequestParam(defaultValue = "9223372036854775807") long beforeRevision,
                                                      @RequestParam(defaultValue = "40") int limit) {
        return CommonResult.success(workspace.checkpoints(canvasId, actors.current(), beforeRevision, limit));
    }

    @PostMapping("/checkpoints")
    public CommonResult<CanvasCommandReceipt> checkpoint(@PathVariable String canvasId,
                                                         @Valid @RequestBody CheckpointCreate command) {
        return CommonResult.success(workspace.checkpoint(canvasId, command, actors.current()));
    }

    @GetMapping("/checkpoints/{checkpointId}")
    public CommonResult<CheckpointDetail> checkpoint(@PathVariable String canvasId, @PathVariable String checkpointId) {
        return CommonResult.success(workspace.checkpoint(canvasId, checkpointId, actors.current()));
    }

    @GetMapping("/checkpoints/{checkpointId}/restore-preview")
    public CommonResult<CanvasCheckpointService.Plan> compare(@PathVariable String canvasId,
                                                              @PathVariable String checkpointId) {
        return CommonResult.success(checkpoints.preview(canvasId, checkpointId, actors.current()));
    }

    @PostMapping("/checkpoints/{checkpointId}/restore")
    public CommonResult<CanvasCommandReceipt> restore(@PathVariable String canvasId, @PathVariable String checkpointId,
                                                      @Valid @RequestBody CanvasCheckpointService.Restore request) {
        return CommonResult.success(checkpoints.restore(canvasId, checkpointId, request, actors.current()));
    }

    @PostMapping("/checkpoints/{checkpointId}/fork")
    public CommonResult<CanvasDocument> fork(@PathVariable String canvasId, @PathVariable String checkpointId,
                                             @Valid @RequestBody CanvasCheckpointForkService.Fork request) {
        return CommonResult.success(checkpointForks.fork(canvasId, checkpointId, request, actors.current()));
    }
}
