package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Contract;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpReviewService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasAgentMcp {
    private final CanvasAgentAccess access;
    private final CanvasMcpContracts contracts;
    private final CanvasMcpReviewService reviews;
    private final CanvasRunRepository runs;
    private final CanvasJson json;

    public Object catalog(ObjectNode input, ToolExecutionContext context) {
        var session = access.require(context);
        long userId = session.context().actor().userId();
        if (input.hasNonNull("contractId")) {
            var contract = contracts.require(input.path("contractId").asText(), userId, false);
            return Map.of("status", "success", "operation", summary(contract), "details",
                    CanvasAgentLibrary.page(json.write(contract.definition()), input.path("offset").asInt(0),
                            input.path("limit").asInt(8000)));
        }
        int page = input.path("page").asInt(0);
        if (page < 0 || page > 100) {
            throw new BusinessException(400, "MCP 操作页码无效");
        }
        var items = contracts.list(userId).stream().filter(Contract::enabled).toList();
        return Map.of("status", "success", "page", page, "items",
                items.stream().skip((long) page * 20).limit(20).map(this::summary).toList(),
                "hasMore", (long) (page + 1) * 20 < items.size());
    }

    private Object summary(Contract contract) {
        return Map.of("id", contract.id(), "name", contract.name(), "version", contract.version(), "enabled",
                contract.enabled(),
                "serverName", contract.definition().tool().serverName(), "toolName",
                contract.definition().tool().toolName(),
                "effect", contract.definition().effect(), "nodeKind", "mcpTool", "config",
                Map.of("contractId", contract.id(), "contractVersion", contract.version(), "inputMode", "NONE",
                        "argumentsJson", "{}"));
    }

    public Object execution(ObjectNode input, ToolExecutionContext context) {
        var session = access.require(context);
        String runId = input.path("canvasRunId").asText();
        if (!runs.run(runId, false).canvasId().equals(session.scope().canvasId())) {
            throw new BusinessException(403, "MCP 调用不属于当前画布");
        }
        var result = reviews.detail(runId, input.path("nodeRunId").asText(), session.context().actor());
        return Map.of("status", "success", "state", result.status(), "revision", result.revision(), "callId",
                result.id(),
                "details", CanvasAgentLibrary.page(json.write(result), input.path("offset").asInt(0),
                        input.path("limit").asInt(8000)));
    }
}
