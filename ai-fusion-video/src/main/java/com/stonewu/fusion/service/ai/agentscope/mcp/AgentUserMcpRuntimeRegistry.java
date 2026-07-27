package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.ai.vo.AgentMcpTestRespVO;
import com.stonewu.fusion.entity.ai.AgentMcpServer;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolManifest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolkitResources;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.mcp.McpClientBuilder;
import io.agentscope.core.tool.mcp.McpClientWrapper;
import io.agentscope.core.tool.mcp.McpTool;
import io.modelcontextprotocol.spec.McpSchema;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
@Slf4j
public class AgentUserMcpRuntimeRegistry implements DisposableBean {

    private static final int MAX_CACHED_USERS = 128;
    private static final int MAX_TOOLS_PER_USER = 256;

    private final AgentMcpServerService serverService;
    private final ObjectMapper objectMapper;
    private final Map<Long, UserSnapshot> snapshots = new LinkedHashMap<>(16, 0.75f, true);

    public List<AgentKernelToolManifest> manifests(long userId) {
        return snapshot(userId).tools().values().stream()
                .map(UserTool::manifest)
                .toList();
    }

    public boolean isTool(long userId, String toolName) {
        return snapshot(userId).tools().containsKey(toolName);
    }

    public List<AgentScopeMcpRegistry.McpToolReference> catalog(long userId) {
        return snapshot(userId).tools().entrySet().stream()
                .map(entry -> new AgentScopeMcpRegistry.McpToolReference(
                        entry.getValue().serverName(),
                        entry.getKey(),
                        entry.getValue().description(),
                        entry.getValue().manifest().readOnly()))
                .sorted(Comparator
                        .comparing(AgentScopeMcpRegistry.McpToolReference::serverName)
                        .thenComparing(AgentScopeMcpRegistry.McpToolReference::toolName))
                .toList();
    }

    public AgentKernelToolkitResources register(long userId, Set<String> whitelist, Toolkit toolkit) {
        UserSnapshot snapshot = snapshot(userId);
        Map<String, List<String>> selected = new LinkedHashMap<>();
        for (String toolName : whitelist) {
            UserTool tool = snapshot.tools().get(toolName);
            if (tool != null) {
                selected.computeIfAbsent(tool.serverName(), ignored -> new ArrayList<>()).add(toolName);
            }
        }
        List<McpClientWrapper> ownedClients = new ArrayList<>();
        try {
            for (Map.Entry<String, List<String>> selection : selected.entrySet()) {
                AgentMcpServerService.AgentScopeV2ServerConfig config =
                        snapshot.servers().get(selection.getKey());
                if (config == null) {
                    throw new IllegalStateException("用户 MCP 服务配置不存在: " + selection.getKey());
                }
                McpClientWrapper client = createClient(config);
                discover(client, config);
                ownedClients.add(client);
                toolkit.registration()
                        .mcpClient(client)
                        .enableTools(selection.getValue())
                        .apply();
            }
            return () -> closeQuietly(ownedClients);
        } catch (RuntimeException failure) {
            closeQuietly(ownedClients);
            throw failure;
        }
    }

    public String configurationFingerprint(long userId) {
        return serverService.configurationFingerprint(userId);
    }

    public AgentMcpTestRespVO test(AgentMcpServer server) {
        AgentMcpServerService.AgentScopeV2ServerConfig config = serverService.runtimeConfig(server);
        McpClientWrapper client = null;
        try {
            client = createClient(config);
            List<McpSchema.Tool> tools = discover(client, config);
            List<AgentMcpTestRespVO.Tool> result = tools.stream()
                    .map(tool -> new AgentMcpTestRespVO.Tool(
                            tool.name(),
                            tool.description() == null ? "" : tool.description(),
                            tool.annotations() != null
                                    && Boolean.TRUE.equals(tool.annotations().readOnlyHint())))
                    .toList();
            return new AgentMcpTestRespVO(
                    true, "连接成功，发现 " + result.size() + " 个工具", result);
        } catch (Exception failure) {
            return new AgentMcpTestRespVO(
                    false,
                    failure.getMessage() == null ? "MCP 连接失败" : failure.getMessage(),
                    List.of());
        } finally {
            closeQuietly(client);
        }
    }

    public synchronized void invalidate(long userId) {
        snapshots.remove(userId);
    }

    @Override
    public synchronized void destroy() {
        snapshots.clear();
    }

    private synchronized UserSnapshot snapshot(long userId) {
        UserSnapshot existing = snapshots.get(userId);
        if (existing != null) {
            return existing;
        }
        UserSnapshot created = connect(userId);
        snapshots.put(userId, created);
        if (snapshots.size() > MAX_CACHED_USERS) {
            Iterator<Map.Entry<Long, UserSnapshot>> iterator = snapshots.entrySet().iterator();
            if (iterator.hasNext()) {
                iterator.next();
                iterator.remove();
            }
        }
        return created;
    }

    private UserSnapshot connect(long userId) {
        Map<String, AgentMcpServerService.AgentScopeV2ServerConfig> servers = new LinkedHashMap<>();
        Map<String, UserTool> tools = new LinkedHashMap<>();
        for (AgentMcpServer server : serverService.listEnabled(userId)) {
            McpClientWrapper client = null;
            try {
                AgentMcpServerService.AgentScopeV2ServerConfig config =
                        serverService.runtimeConfig(server);
                client = createClient(config);
                List<McpSchema.Tool> discovered = discover(client, config);
                Map<String, UserTool> serverTools = describe(config, discovered, tools.keySet());
                if (tools.size() + serverTools.size() > MAX_TOOLS_PER_USER) {
                    throw new IllegalStateException("每个用户最多启用 256 个 MCP 工具");
                }
                tools.putAll(serverTools);
                servers.put(server.getName(), config);
            } catch (Exception failure) {
                log.warn("跳过不可用的用户 MCP 服务 [userId={}, server={}]: {}",
                        userId, server.getName(), failure.getMessage());
            } finally {
                closeQuietly(client);
            }
        }
        return new UserSnapshot(Map.copyOf(servers), Map.copyOf(tools));
    }

    private Map<String, UserTool> describe(
            AgentMcpServerService.AgentScopeV2ServerConfig config,
            List<McpSchema.Tool> discovered,
            Set<String> existingNames) throws Exception {
        Map<String, McpSchema.Tool> byName = new LinkedHashMap<>();
        for (McpSchema.Tool tool : discovered) {
            if (tool.name() == null || tool.name().isBlank()) {
                throw new IllegalStateException("MCP 服务返回了空工具名");
            }
            if (byName.put(tool.name(), tool) != null) {
                throw new IllegalStateException("MCP 服务返回重复工具名: " + tool.name());
            }
        }
        Set<String> selected = config.enabledTools().isEmpty()
                ? new LinkedHashSet<>(byName.keySet())
                : new LinkedHashSet<>(config.enabledTools());
        if (!byName.keySet().containsAll(selected)) {
            Set<String> unavailable = new LinkedHashSet<>(selected);
            unavailable.removeAll(byName.keySet());
            throw new IllegalStateException("配置的 MCP 工具不存在: " + unavailable);
        }
        Set<String> duplicates = new LinkedHashSet<>(selected);
        duplicates.retainAll(existingNames);
        if (!duplicates.isEmpty()) {
            throw new IllegalStateException("多个 MCP 服务暴露了同名工具: " + duplicates);
        }
        Map<String, UserTool> result = new LinkedHashMap<>();
        for (String name : selected) {
            McpSchema.Tool tool = byName.get(name);
            result.put(name, new UserTool(
                    config.name(),
                    tool.description() == null ? "" : tool.description(),
                    manifest(tool)));
        }
        return result;
    }

    private AgentKernelToolManifest manifest(McpSchema.Tool tool) throws Exception {
        Map<String, Object> parameters = McpTool.convertMcpSchemaToParameters(
                tool.inputSchema(), Set.of());
        AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                objectMapper,
                objectMapper.writeValueAsString(parameters),
                tool.name());
        boolean readOnly = tool.annotations() != null
                && Boolean.TRUE.equals(tool.annotations().readOnlyHint());
        return new AgentKernelToolManifest(
                tool.name(),
                AgentKernelToolManifest.schemaSha256(schema.canonicalJson()),
                readOnly,
                false);
    }

    private McpClientWrapper createClient(
            AgentMcpServerService.AgentScopeV2ServerConfig config) {
        McpClientBuilder builder = McpClientBuilder.create(config.name());
        if ("sse".equals(config.transport())) {
            builder.sseTransport(config.url());
        } else {
            builder.streamableHttpTransport(config.url());
        }
        if (!config.headers().isEmpty()) {
            builder.headers(config.headers());
        }
        if (!config.queryParams().isEmpty()) {
            builder.queryParams(config.queryParams());
        }
        if (!config.protocolVersions().isEmpty()) {
            builder.protocolVersions(config.protocolVersions().toArray(String[]::new));
        }
        return builder.timeout(config.timeout())
                .initializationTimeout(config.initializationTimeout())
                .buildAsync()
                .block(config.initializationTimeout());
    }

    private List<McpSchema.Tool> discover(
            McpClientWrapper client,
            AgentMcpServerService.AgentScopeV2ServerConfig config) {
        List<McpSchema.Tool> tools = client.initialize()
                .then(client.listTools())
                .block(config.initializationTimeout().plus(config.timeout()));
        if (tools == null) {
            throw new IllegalStateException("MCP 服务没有返回工具列表");
        }
        return List.copyOf(tools);
    }

    private void closeQuietly(McpClientWrapper client) {
        if (client == null) {
            return;
        }
        try {
            client.close();
        } catch (RuntimeException ignored) {
            // Preserve the original connection failure.
        }
    }

    private void closeQuietly(Iterable<McpClientWrapper> clients) {
        clients.forEach(this::closeQuietly);
    }

    private record UserSnapshot(
            Map<String, AgentMcpServerService.AgentScopeV2ServerConfig> servers,
            Map<String, UserTool> tools) {
    }

    private record UserTool(
            String serverName,
            String description,
            AgentKernelToolManifest manifest) {
    }
}
