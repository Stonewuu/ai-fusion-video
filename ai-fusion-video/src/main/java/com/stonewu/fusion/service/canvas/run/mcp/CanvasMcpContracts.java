package com.stonewu.fusion.service.canvas.run.mcp;

import com.stonewu.fusion.mapper.canvas.CanvasMcpContractMapper;
import com.stonewu.fusion.entity.canvas.CanvasMcpContractEntity;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpWorkflowTool;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;

import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.List;
import java.util.Objects;
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.*;

@Service
@RequiredArgsConstructor
public class CanvasMcpContracts {
    public record Save(String id, Long expectedVersion, @NotBlank @Size(max = 128) String name,
                       @NotBlank String toolName, @NotBlank String schemaHash, @NotBlank String configurationHash,
                       @NotNull Effect effect, @NotNull Idempotency idempotency, String idempotencyParameter,
                       @NotNull OutputMode outputMode, @NotNull JsonNode outputSchema,
                       int timeoutSeconds, boolean enabled) {
    }

    private final CanvasMcpContractMapper contracts;
    private final UserMapper users;
    private final CanvasJson json;
    private final AgentScopeMcpRegistry tools;
    private final CanvasMcpSchemas schemas;
    private final TransactionTemplate transactions;
    private static final String AGENT_TYPE = "canvas_assistant";

    public List<McpWorkflowTool> tools(long userId) {
        return tools.workflowTools(AGENT_TYPE, userId);
    }

    public List<Contract> list(long userId) {
        return contracts.selectList(
                        Wrappers.<CanvasMcpContractEntity>lambdaQuery().select(CanvasMcpContractEntity::getId)
                                .eq(CanvasMcpContractEntity::getUserId, userId)
                                .orderByAsc(CanvasMcpContractEntity::getName, CanvasMcpContractEntity::getId))
                .stream().map(CanvasMcpContractEntity::getId).toList()
                .stream().map(id -> require(id, userId, false)).toList();
    }

    @Cacheable(value = "canvasMcpContract", key = "#userId + ':' + #id + ':' + #version")
    public Contract version(String id, long version, long userId) {
        var result = require(id, userId, false);
        if (result.version() != version) {
            throw CanvasRunException.conflict("MCP 操作契约已更新，请重新选择当前版本");
        }
        return result;
    }

    public Contract require(String id, long userId, boolean lock) {
        var row = lock ? contracts.lock(id, userId) : contracts.selectOne(
                Wrappers.<CanvasMcpContractEntity>lambdaQuery()
                        .eq(CanvasMcpContractEntity::getId, id).eq(CanvasMcpContractEntity::getUserId, userId));
        if (row == null) {
            throw new CanvasRunException(404, "MCP_CONTRACT_NOT_FOUND", "MCP 操作不存在或不属于当前账号");
        }
        return new Contract(row.getId(), row.getUserId(), row.getName(), row.getVersion(), row.getEnabled(),
                json.read(row.getDefinitionJson(), Definition.class), row.getDefinitionHash());
    }

    @CacheEvict(value = "canvasMcpContract", allEntries = true)
    public Contract save(long userId, Save request) {
        if (request.name() == null || request.name().isBlank() || request.name().length() > 128
                || request.timeoutSeconds() < 1 || request.timeoutSeconds() > 600) {
            throw CanvasRunException.invalid("请填写操作名称和 1–600 秒的执行超时");
        }
        var tool = tools(userId).stream().filter(value -> value.toolName().equals(request.toolName())).findFirst()
                .orElseThrow(() -> CanvasRunException.invalid("此 MCP 工具未配置或不可用于画布"));
        if (!tool.schemaHash().equals(request.schemaHash()) || !tool.configurationHash()
                .equals(request.configurationHash())) {
            throw CanvasRunException.conflict("MCP 定义或服务配置已变化，请重新加载工具");
        }
        schemas.compile(tool.inputSchema());
        schemas.compile(request.outputSchema());
        if (request.effect() == null || request.idempotency() == null || request.outputMode() == null) {
            throw CanvasRunException.invalid("请明确操作副作用、幂等方式和输出类型");
        }
        if (request.effect() == Effect.READ_ONLY && !tool.readOnly()) {
            throw CanvasRunException.invalid("远端未声明只读，此工具不能登记为只读操作");
        }
        if (tool.destructive() && request.effect() != Effect.DESTRUCTIVE) {
            throw CanvasRunException.invalid("远端未排除破坏性副作用，请按危险操作登记");
        }
        if ((request.effect() == Effect.READ_ONLY) != (request.idempotency() == Idempotency.READ_ONLY)) {
            throw CanvasRunException.invalid("只读与写入操作的幂等声明不一致");
        }
        String parameter = request.idempotencyParameter() == null ? "" : request.idempotencyParameter().trim();
        if (request.idempotency() == Idempotency.KEYED && (parameter.isBlank()
                || !tool.inputSchema().path("properties").path(parameter).path("type").asText().equals("string"))) {
            throw CanvasRunException.invalid("幂等键必须是工具已声明的顶层字符串参数");
        }
        if (request.idempotency() != Idempotency.KEYED && !parameter.isBlank()) {
            throw CanvasRunException.invalid("只有按幂等键执行的操作可以指定幂等参数");
        }
        if (request.outputMode() == OutputMode.TEXT && !request.outputSchema().path("type").asText().equals("string")) {
            throw CanvasRunException.invalid("纯文本输出的契约必须声明 string 类型");
        }
        if (request.outputMode() == OutputMode.STRUCTURED_JSON && (tool.outputSchema() == null || tool.outputSchema()
                .isNull())) {
            throw CanvasRunException.invalid("远端未声明结构化输出，请选择实际返回的文本类型");
        }
        var definition = new Definition(tool, request.effect(), request.idempotency(), parameter, request.outputMode(),
                request.outputSchema().deepCopy(), request.timeoutSeconds());
        String id = request.id() == null ? UUID.randomUUID().toString() : request.id();
        return transactions.execute(status -> {
            if (users.lockEnabled(userId) == null) {
                throw new CanvasRunException(403, "MCP_OWNER_UNAVAILABLE", "当前账号已不可用");
            }
            if (request.id() == null) {
                if (list(userId).size() >= 128) {
                    throw CanvasRunException.invalid("每个账号最多登记 128 个 MCP 工作流操作");
                }
                contracts.create(id, userId, request.name().trim(), tool.toolName(), tool.serverName(),
                        json.write(definition), json.hash(definition), request.enabled());
            } else {
                var current = require(id, userId, true);
                if (request.expectedVersion() == null || request.expectedVersion() != current.version()) {
                    throw CanvasRunException.conflict("MCP 契约已被其他操作修改");
                }
                contracts.revise(request.name().trim(), tool.toolName(), tool.serverName(), json.write(definition),
                        json.hash(definition), request.enabled(), id);
            }
            return require(id, userId, false);
        });
    }

    @CacheEvict(value = "canvasMcpContract", allEntries = true)
    public void remove(String id, long version, long userId) {
        transactions.executeWithoutResult(status -> {
            var current = require(id, userId, true);
            if (current.version() != version) {
                throw CanvasRunException.conflict("MCP 契约已变化，请刷新");
            }
            contracts.remove(id);
        });
    }

    public ObjectNode freeze(CanvasNode node, CanvasActor actor) {
        var contract = require(node.config().path("contractId").asText(), actor.userId(), false);
        verify(contract, actor.userId());
        if (!contract.enabled() || !node.config().has("contractVersion") || contract.version() != node.config()
                .path("contractVersion").asLong()) {
            throw CanvasRunException.conflict("MCP 操作停用或版本已变化，请重新选择操作");
        }
        return json.object(contract);
    }

    public void verify(Contract frozen, long userId) {
        var current = require(frozen.id(), userId, false);
        if (!current.enabled() || !Objects.equals(current.hash(),
                frozen.hash()) || current.version() != frozen.version()
                || frozen.userId() != userId) {
            throw CanvasRunException.conflict("MCP 操作在计划确认后变化，未发送请求");
        }
        if (!frozen.definition().tool().configurationHash()
                .equals(tools.workflowConfigurationHash(AGENT_TYPE, userId, frozen.definition().tool().toolName()))) {
            throw CanvasRunException.conflict("MCP 服务配置已变化，请重新登记操作契约");
        }
    }

    public McpWorkflowTool.Opened open(Contract frozen, long userId) {
        verify(frozen, userId);
        var opened = tools.openWorkflowTool(AGENT_TYPE, userId, frozen.definition().tool().toolName());
        var actual = opened.definition();
        var expected = frozen.definition().tool();
        if (!actual.schemaHash().equals(expected.schemaHash()) || !actual.configurationHash()
                .equals(expected.configurationHash())
                || !actual.serverName().equals(expected.serverName())) {
            opened.close();
            throw CanvasRunException.conflict("远端 MCP 工具或连接配置已变化，请重新登记并确认计划");
        }
        return opened;
    }
}
