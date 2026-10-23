package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.service.ai.configuration.AgentSkillSelection;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.CanvasEventStream;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentProfiles;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.capability.CanvasNodeCapability;
import com.stonewu.fusion.service.canvas.capability.CanvasWritebackFields;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.model.*;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.util.List;

@RestController
@RequestMapping("/api/canvases")
@RequiredArgsConstructor
public class CanvasController {
    private final CanvasActorResolver actors;
    private final CanvasCommandService commands;
    private final CanvasQueryService queries;
    private final CanvasCapabilityRegistry capabilities;
    private final CanvasEventStream events;
    private final CanvasModelCatalog models;
    private final CanvasAgentProfiles profiles;
    private final AgentSkillSelection skills;

    public record Catalog(int schemaVersion, List<CanvasNodeCapability> nodes, List<CanvasModelCatalog.Option> models,
                          List<CanvasModelCatalog.AudioService> audioServices,
                          List<CanvasWritebackFields.Field> writebackFields,
                          List<CanvasAgentProfiles.Profile> agentProfiles, List<AgentSkillSelection.Option> skills) {
    }

    @GetMapping("/capabilities")
    public CommonResult<Catalog> capabilities() {
        var actor = actors.current();
        return CommonResult.success(
                new Catalog(CanvasCapabilityRegistry.SCHEMA_VERSION, capabilities.list(), models.list(),
                        models.audioServices(), CanvasWritebackFields.list(),
                        profiles.list(actor.userId()), skills.catalog(actor.userId())));
    }

    @GetMapping
    public CommonResult<PageResult<CanvasListItem>> list(@RequestParam(required = false) Long projectId,
                                                         @RequestParam(defaultValue = "") String search,
                                                         @RequestParam(defaultValue = "false") boolean archived,
                                                         @RequestParam(defaultValue = "1") int pageNo,
                                                         @RequestParam(defaultValue = "30") int pageSize) {
        return CommonResult.success(queries.list(actors.current(), projectId, search, archived, pageNo, pageSize));
    }

    @PostMapping
    public CommonResult<CanvasDocument> create(@Valid @RequestBody CanvasCreateCommand command) {
        return CommonResult.success(commands.create(command, actors.current()));
    }

    @GetMapping("/{id}")
    public CommonResult<CanvasSnapshot> snapshot(@PathVariable String id) {
        return CommonResult.success(queries.snapshot(id, actors.current()));
    }

    @PostMapping("/{id}/commands")
    public CommonResult<CanvasCommandReceipt> edit(@PathVariable String id,
                                                   @Valid @RequestBody CanvasEditCommand command) {
        return CommonResult.success(commands.edit(id, command, actors.current()));
    }

    @GetMapping("/{id}/events")
    public CommonResult<CanvasEventPage> history(@PathVariable String id,
                                                 @RequestParam(defaultValue = "0") long afterSequence,
                                                 @RequestParam(defaultValue = "200") int limit) {
        return CommonResult.success(queries.events(id, actors.current(), afterSequence, limit));
    }

    @GetMapping(value = "/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<Object>> stream(@PathVariable String id,
                                                @RequestParam(defaultValue = "0") long afterSequence,
                                                @RequestHeader(value = "Last-Event-ID", required = false) String lastEventId) {
        long cursor = afterSequence;
        if (lastEventId != null) {
            try {
                cursor = Long.parseLong(lastEventId);
            } catch (NumberFormatException invalid) {
                throw new CanvasException(CanvasErrorCode.INVALID_CURSOR, "事件游标无效");
            }
        }
        return events.open(id, actors.current(), cursor);
    }
}
