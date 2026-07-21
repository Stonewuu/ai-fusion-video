package com.stonewu.fusion.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;
import java.util.Objects;

@ConfigurationProperties(prefix = "fusion.agentscope.v2")
public class AgentScopeV2Properties {

    private Cache cache = new Cache();
    private State state = new State();
    private Ingress ingress = new Ingress();
    private Execution execution = new Execution();

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
        private Mode mode = Mode.REDIS;
        private String keyPrefix = "afv:agentscope:v2:";

        public Mode getMode() {
            return mode;
        }

        public void setMode(Mode mode) {
            this.mode = Objects.requireNonNull(mode, "mode must not be null");
        }

        public String getKeyPrefix() {
            return keyPrefix;
        }

        public void setKeyPrefix(String keyPrefix) {
            if (keyPrefix == null || keyPrefix.isBlank()) {
                throw new IllegalArgumentException("keyPrefix must not be blank");
            }
            this.keyPrefix = keyPrefix.trim();
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
    }

    public enum Mode {
        IN_MEMORY,
        REDIS
    }

    private static Duration requirePositive(Duration value, String name) {
        Objects.requireNonNull(value, name + " must not be null");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(name + " must be greater than zero");
        }
        return value;
    }
}
