package com.stonewu.fusion.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

@ConfigurationProperties(prefix = "fusion.agentscope.v2")
public class AgentScopeV2Properties {

    private Cache cache = new Cache();
    private State state = new State();
    private Ingress ingress = new Ingress();
    private Execution execution = new Execution();
    private Skills skills = new Skills();
    private Mcp mcp = new Mcp();

    public Cache getCache() {
        return cache;
    }

    public void setCache(Cache cache) {
        this.cache = Objects.requireNonNull(cache, "cache must not be null");
    }

    public State getState() {
        return state;
    }

    public void setState(State state) {
        this.state = Objects.requireNonNull(state, "state must not be null");
    }

    public Ingress getIngress() {
        return ingress;
    }

    public void setIngress(Ingress ingress) {
        this.ingress = Objects.requireNonNull(ingress, "ingress must not be null");
    }

    public Execution getExecution() {
        return execution;
    }

    public void setExecution(Execution execution) {
        this.execution = Objects.requireNonNull(execution, "execution must not be null");
    }

    public Skills getSkills() {
        return skills;
    }

    public void setSkills(Skills skills) {
        this.skills = Objects.requireNonNull(skills, "skills must not be null");
    }

    public Mcp getMcp() {
        return mcp;
    }

    public void setMcp(Mcp mcp) {
        this.mcp = Objects.requireNonNull(mcp, "mcp must not be null");
    }

    public static final class Cache {
        private int maximumSize = 64;
        private Duration expireAfterAccess = Duration.ofMinutes(30);
        private Duration capacityWait = Duration.ofSeconds(5);

        public int getMaximumSize() {
            return maximumSize;
        }

        public void setMaximumSize(int maximumSize) {
            if (maximumSize <= 0) {
                throw new IllegalArgumentException("maximumSize must be greater than zero");
            }
            this.maximumSize = maximumSize;
        }

        public Duration getExpireAfterAccess() {
            return expireAfterAccess;
        }

        public void setExpireAfterAccess(Duration expireAfterAccess) {
            this.expireAfterAccess = requirePositive(expireAfterAccess, "expireAfterAccess");
        }

        public Duration getCapacityWait() {
            return capacityWait;
        }

        public void setCapacityWait(Duration capacityWait) {
            this.capacityWait = requirePositive(capacityWait, "capacityWait");
        }
    }

    public static final class State {
        private Mode mode = Mode.MYSQL;
        private String databaseName = "ai_fusion_video";
        private String tableName = "afv_agent_state";

        public Mode getMode() {
            return mode;
        }

        public void setMode(Mode mode) {
            this.mode = Objects.requireNonNull(mode, "mode must not be null");
        }

        public String getDatabaseName() {
            return databaseName;
        }

        public void setDatabaseName(String databaseName) {
            if (databaseName == null || databaseName.isBlank()) {
                throw new IllegalArgumentException("databaseName must not be blank");
            }
            this.databaseName = databaseName.trim();
        }

        public String getTableName() {
            return tableName;
        }

        public void setTableName(String tableName) {
            if (tableName == null || tableName.isBlank()) {
                throw new IllegalArgumentException("tableName must not be blank");
            }
            this.tableName = tableName.trim();
        }
    }

    public static final class Ingress {
        private int maxEvents = 4096;
        private long maxBytes = 8L * 1024L * 1024L;
        private Duration coalesceDelay = Duration.ofMillis(50);
        private int coalesceMaxChars = 1024;

        public int getMaxEvents() {
            return maxEvents;
        }

        public void setMaxEvents(int maxEvents) {
            if (maxEvents <= 0) {
                throw new IllegalArgumentException("maxEvents must be greater than zero");
            }
            this.maxEvents = maxEvents;
        }

        public long getMaxBytes() {
            return maxBytes;
        }

        public void setMaxBytes(long maxBytes) {
            if (maxBytes <= 0) {
                throw new IllegalArgumentException("maxBytes must be greater than zero");
            }
            this.maxBytes = maxBytes;
        }

        public Duration getCoalesceDelay() {
            return coalesceDelay;
        }

        public void setCoalesceDelay(Duration coalesceDelay) {
            this.coalesceDelay = requirePositive(coalesceDelay, "coalesceDelay");
        }

        public int getCoalesceMaxChars() {
            return coalesceMaxChars;
        }

        public void setCoalesceMaxChars(int coalesceMaxChars) {
            if (coalesceMaxChars <= 0) {
                throw new IllegalArgumentException("coalesceMaxChars must be greater than zero");
            }
            this.coalesceMaxChars = coalesceMaxChars;
        }
    }

    public static final class Execution {
        private String instanceId;
        private Duration ownerLease = Duration.ofSeconds(30);
        private Duration runTimeout = Duration.ofMinutes(30);
        private int maxIters = 999;

        public String getInstanceId() {
            return instanceId;
        }

        public void setInstanceId(String instanceId) {
            this.instanceId = instanceId == null || instanceId.isBlank()
                    ? null
                    : instanceId.trim();
        }

        public Duration getOwnerLease() {
            return ownerLease;
        }

        public void setOwnerLease(Duration ownerLease) {
            this.ownerLease = requirePositive(ownerLease, "ownerLease");
        }

        public Duration getRunTimeout() {
            return runTimeout;
        }

        public void setRunTimeout(Duration runTimeout) {
            this.runTimeout = requirePositive(runTimeout, "runTimeout");
        }

        public int getMaxIters() {
            return maxIters;
        }

        public void setMaxIters(int maxIters) {
            if (maxIters <= 0) {
                throw new IllegalArgumentException("maxIters must be greater than zero");
            }
            this.maxIters = maxIters;
        }
    }

    public static final class Skills {
        private boolean enabled;
        private boolean failFast = true;
        private Map<String, SkillRepository> repositories = new LinkedHashMap<>();
        private Map<String, String> displayNames = new LinkedHashMap<>();

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public boolean isFailFast() {
            return failFast;
        }

        public void setFailFast(boolean failFast) {
            this.failFast = failFast;
        }

        public Map<String, SkillRepository> getRepositories() {
            return repositories;
        }

        public void setRepositories(Map<String, SkillRepository> repositories) {
            this.repositories = repositories == null
                    ? new LinkedHashMap<>()
                    : new LinkedHashMap<>(repositories);
        }

        public Map<String, String> getDisplayNames() {
            return displayNames;
        }

        public void setDisplayNames(Map<String, String> displayNames) {
            this.displayNames = displayNames == null
                    ? new LinkedHashMap<>()
                    : new LinkedHashMap<>(displayNames);
        }
    }

    public static final class SkillRepository {
        private boolean enabled = true;
        private String location;
        private boolean lazy;

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public String getLocation() {
            return location;
        }

        public void setLocation(String location) {
            this.location = normalize(location);
        }

        public boolean isLazy() {
            return lazy;
        }

        public void setLazy(boolean lazy) {
            this.lazy = lazy;
        }
    }

    public static final class Mcp {
        private boolean enabled;
        private boolean failFast = true;
        private Map<String, McpServer> servers = new LinkedHashMap<>();

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public boolean isFailFast() {
            return failFast;
        }

        public void setFailFast(boolean failFast) {
            this.failFast = failFast;
        }

        public Map<String, McpServer> getServers() {
            return servers;
        }

        public void setServers(Map<String, McpServer> servers) {
            this.servers = servers == null
                    ? new LinkedHashMap<>()
                    : new LinkedHashMap<>(servers);
        }
    }

    public static final class McpServer {
        private boolean enabled = true;
        private String transport;
        private String command;
        private List<String> args = new ArrayList<>();
        private Map<String, String> env = new LinkedHashMap<>();
        private String url;
        private Map<String, String> headers = new LinkedHashMap<>();
        private Map<String, String> queryParams = new LinkedHashMap<>();
        private List<String> enabledTools = new ArrayList<>();
        private Set<String> agentTypes = new LinkedHashSet<>();
        private List<String> protocolVersions = new ArrayList<>();
        private Duration timeout = Duration.ofSeconds(120);
        private Duration initializationTimeout = Duration.ofSeconds(30);

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public String getTransport() {
            return transport;
        }

        public void setTransport(String transport) {
            this.transport = normalize(transport);
        }

        public String getCommand() {
            return command;
        }

        public void setCommand(String command) {
            this.command = normalize(command);
        }

        public List<String> getArgs() {
            return args;
        }

        public void setArgs(List<String> args) {
            this.args = args == null ? new ArrayList<>() : new ArrayList<>(args);
        }

        public Map<String, String> getEnv() {
            return env;
        }

        public void setEnv(Map<String, String> env) {
            this.env = env == null ? new LinkedHashMap<>() : new LinkedHashMap<>(env);
        }

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = normalize(url);
        }

        public Map<String, String> getHeaders() {
            return headers;
        }

        public void setHeaders(Map<String, String> headers) {
            this.headers = headers == null
                    ? new LinkedHashMap<>()
                    : new LinkedHashMap<>(headers);
        }

        public Map<String, String> getQueryParams() {
            return queryParams;
        }

        public void setQueryParams(Map<String, String> queryParams) {
            this.queryParams = queryParams == null
                    ? new LinkedHashMap<>()
                    : new LinkedHashMap<>(queryParams);
        }

        public List<String> getEnabledTools() {
            return enabledTools;
        }

        public void setEnabledTools(List<String> enabledTools) {
            this.enabledTools = normalizedList(enabledTools);
        }

        public Set<String> getAgentTypes() {
            return agentTypes;
        }

        public void setAgentTypes(Set<String> agentTypes) {
            this.agentTypes = new LinkedHashSet<>(normalizedList(
                    agentTypes == null ? List.of() : new ArrayList<>(agentTypes)));
        }

        public List<String> getProtocolVersions() {
            return protocolVersions;
        }

        public void setProtocolVersions(List<String> protocolVersions) {
            this.protocolVersions = normalizedList(protocolVersions);
        }

        public Duration getTimeout() {
            return timeout;
        }

        public void setTimeout(Duration timeout) {
            this.timeout = requirePositive(timeout, "mcp server timeout");
        }

        public Duration getInitializationTimeout() {
            return initializationTimeout;
        }

        public void setInitializationTimeout(Duration initializationTimeout) {
            this.initializationTimeout = requirePositive(
                    initializationTimeout, "mcp server initializationTimeout");
        }
    }

    public enum Mode {
        IN_MEMORY,
        MYSQL
    }

    private static Duration requirePositive(Duration value, String name) {
        Objects.requireNonNull(value, name + " must not be null");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(name + " must be greater than zero");
        }
        return value;
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private static List<String> normalizedList(List<String> values) {
        if (values == null || values.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> normalized = new ArrayList<>(values.size());
        for (String value : values) {
            String item = normalize(value);
            if (item != null && !normalized.contains(item)) {
                normalized.add(item);
            }
        }
        return normalized;
    }
}
