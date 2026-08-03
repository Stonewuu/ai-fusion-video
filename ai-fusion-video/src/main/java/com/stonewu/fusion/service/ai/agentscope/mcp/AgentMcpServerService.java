package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AgentMcpServerRespVO;
import com.stonewu.fusion.controller.ai.vo.AgentMcpServerSaveReqVO;
import com.stonewu.fusion.entity.ai.AgentMcpServer;
import com.stonewu.fusion.mapper.ai.AgentMcpServerMapper;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolManifest;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.net.InetAddress;
import java.net.URI;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

@Service
@RequiredArgsConstructor
public class AgentMcpServerService {

    private static final Set<String> TRANSPORTS = Set.of("http", "sse");
    private static final int MAX_SERVERS_PER_USER = 32;
    private static final TypeReference<Map<String, String>> STRING_MAP = new TypeReference<>() {
    };
    private static final TypeReference<List<String>> STRING_LIST = new TypeReference<>() {
    };

    private final AgentMcpServerMapper serverMapper;
    private final ObjectMapper objectMapper;

    public List<AgentMcpServer> list(long userId) {
        return serverMapper.selectList(new LambdaQueryWrapper<AgentMcpServer>()
                .eq(AgentMcpServer::getUserId, userId)
                .orderByAsc(AgentMcpServer::getName));
    }

    public List<AgentMcpServer> listEnabled(long userId) {
        return list(userId).stream()
                .filter(server -> Integer.valueOf(1).equals(server.getStatus()))
                .toList();
    }

    public AgentMcpServer requireOwned(Long id, long userId) {
        AgentMcpServer server = serverMapper.selectById(id);
        if (server == null || !Long.valueOf(userId).equals(server.getUserId())) {
            throw new BusinessException(404, "MCP 服务不存在");
        }
        return server;
    }

    @Transactional
    @CacheEvict(value = "agentMcpServers", key = "#userId")
    public AgentMcpServer save(long userId, AgentMcpServerSaveReqVO request) {
        AgentMcpServer server = request.id() == null
                ? AgentMcpServer.builder().userId(userId).build()
                : requireOwned(request.id(), userId);
        if (request.id() == null && list(userId).size() >= MAX_SERVERS_PER_USER) {
            throw new BusinessException("每个用户最多配置 32 个 MCP 服务");
        }
        String name = requireText(request.name(), "MCP 服务名称");
        boolean duplicate = list(userId).stream()
                .anyMatch(existing -> name.equalsIgnoreCase(existing.getName())
                        && !Objects.equals(existing.getId(), request.id()));
        if (duplicate) {
            throw new BusinessException("同名 MCP 服务已存在");
        }
        String transport = requireText(request.transport(), "MCP 传输方式")
                .toLowerCase(Locale.ROOT);
        if (!TRANSPORTS.contains(transport)) {
            throw new BusinessException("用户自定义 MCP 仅支持 HTTP 或 SSE 传输");
        }
        String url = validatePublicHttpUrl(request.url());
        server.setName(name);
        server.setTransport(transport);
        server.setUrl(url);
        server.setHeadersJson(writeJson(normalizeMap(request.headers())));
        server.setQueryParamsJson(writeJson(normalizeMap(request.queryParams())));
        server.setEnabledToolsJson(writeJson(normalizeList(request.enabledTools())));
        server.setProtocolVersionsJson(writeJson(normalizeList(request.protocolVersions())));
        server.setTimeoutSeconds(request.timeoutSeconds() == null ? 120 : request.timeoutSeconds());
        server.setInitializationTimeoutSeconds(
                request.initializationTimeoutSeconds() == null
                        ? 30
                        : request.initializationTimeoutSeconds());
        server.setStatus(request.status() == null ? 1 : request.status() == 1 ? 1 : 0);
        server.setLastTestStatus(null);
        server.setLastTestMessage(null);
        if (server.getId() == null) {
            serverMapper.insert(server);
        } else {
            serverMapper.updateById(server);
        }
        return server;
    }

    @Transactional
    @CacheEvict(value = "agentMcpServers", key = "#userId")
    public void delete(long userId, Long id) {
        requireOwned(id, userId);
        serverMapper.deleteById(id);
    }

    @Transactional
    @CacheEvict(value = "agentMcpServers", key = "#userId")
    public void recordTest(long userId, Long id, boolean success, String message) {
        AgentMcpServer server = requireOwned(id, userId);
        server.setLastTestStatus(success ? "success" : "failed");
        server.setLastTestMessage(message == null || message.length() <= 1000
                ? message
                : message.substring(0, 1000));
        serverMapper.updateById(server);
    }

    public AgentMcpServerRespVO toResponse(AgentMcpServer server) {
        return new AgentMcpServerRespVO(
                server.getId(),
                server.getName(),
                server.getTransport(),
                server.getUrl(),
                readMap(server.getHeadersJson()),
                readMap(server.getQueryParamsJson()),
                readList(server.getEnabledToolsJson()),
                readList(server.getProtocolVersionsJson()),
                server.getTimeoutSeconds(),
                server.getInitializationTimeoutSeconds(),
                server.getStatus(),
                server.getLastTestStatus(),
                server.getLastTestMessage(),
                server.getUpdateTime());
    }

    public AgentScopeV2ServerConfig runtimeConfig(AgentMcpServer server) {
        return new AgentScopeV2ServerConfig(
                server.getName(),
                server.getTransport(),
                server.getUrl(),
                readMap(server.getHeadersJson()),
                readMap(server.getQueryParamsJson()),
                readList(server.getEnabledToolsJson()),
                readList(server.getProtocolVersionsJson()),
                Duration.ofSeconds(server.getTimeoutSeconds()),
                Duration.ofSeconds(server.getInitializationTimeoutSeconds()));
    }

    public String configurationFingerprint(long userId) {
        StringBuilder canonical = new StringBuilder();
        for (AgentMcpServer server : listEnabled(userId)) {
            append(canonical, server.getId());
            append(canonical, server.getName());
            append(canonical, server.getTransport());
            append(canonical, server.getUrl());
            append(canonical, server.getHeadersJson());
            append(canonical, server.getQueryParamsJson());
            append(canonical, server.getEnabledToolsJson());
            append(canonical, server.getProtocolVersionsJson());
            append(canonical, server.getTimeoutSeconds());
            append(canonical, server.getInitializationTimeoutSeconds());
        }
        return AgentKernelToolManifest.schemaSha256(canonical.toString());
    }

    private String validatePublicHttpUrl(String value) {
        String url = requireText(value, "MCP 地址");
        try {
            URI uri = URI.create(url);
            if (!("http".equalsIgnoreCase(uri.getScheme())
                    || "https".equalsIgnoreCase(uri.getScheme()))
                    || uri.getHost() == null) {
                throw new BusinessException("MCP 地址必须是有效的 HTTP(S) URL");
            }
            if (uri.getUserInfo() != null) {
                throw new BusinessException("MCP 地址不能包含 URL 用户凭据");
            }
            for (InetAddress address : InetAddress.getAllByName(uri.getHost())) {
                if (address.isAnyLocalAddress()
                        || address.isLoopbackAddress()
                        || address.isLinkLocalAddress()
                        || address.isSiteLocalAddress()
                        || address.isMulticastAddress()) {
                    throw new BusinessException("MCP 地址不能指向本机或内网地址");
                }
            }
            return uri.normalize().toString();
        } catch (BusinessException failure) {
            throw failure;
        } catch (Exception failure) {
            throw new BusinessException("无法解析 MCP 地址: " + failure.getMessage());
        }
    }

    private Map<String, String> normalizeMap(Map<String, String> values) {
        Map<String, String> result = new LinkedHashMap<>();
        if (values == null) {
            return result;
        }
        values.forEach((key, value) -> {
            if (key != null && !key.isBlank() && value != null) {
                result.put(key.trim(), value.trim());
            }
        });
        return result;
    }

    private List<String> normalizeList(List<String> values) {
        if (values == null) {
            return List.of();
        }
        return values.stream()
                .filter(Objects::nonNull)
                .map(String::trim)
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
    }

    private String writeJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception failure) {
            throw new BusinessException("MCP 配置序列化失败");
        }
    }

    private Map<String, String> readMap(String value) {
        if (value == null || value.isBlank()) {
            return Map.of();
        }
        try {
            return objectMapper.readValue(value, STRING_MAP);
        } catch (Exception failure) {
            throw new BusinessException("MCP 请求参数配置损坏");
        }
    }

    private List<String> readList(String value) {
        if (value == null || value.isBlank()) {
            return List.of();
        }
        try {
            return objectMapper.readValue(value, STRING_LIST);
        } catch (Exception failure) {
            throw new BusinessException("MCP 工具配置损坏");
        }
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new BusinessException(field + "不能为空");
        }
        return value.trim();
    }

    private void append(StringBuilder target, Object value) {
        String text = value == null ? "" : String.valueOf(value);
        target.append(text.length()).append(':').append(text).append('\n');
    }

    public record AgentScopeV2ServerConfig(
            String name,
            String transport,
            String url,
            Map<String, String> headers,
            Map<String, String> queryParams,
            List<String> enabledTools,
            List<String> protocolVersions,
            Duration timeout,
            Duration initializationTimeout) {
    }
}
