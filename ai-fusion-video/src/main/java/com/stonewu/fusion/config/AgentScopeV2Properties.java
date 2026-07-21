package com.stonewu.fusion.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;
import java.util.Objects;

@ConfigurationProperties(prefix = "fusion.agentscope.v2")
public class AgentScopeV2Properties {

    private Cache cache = new Cache();
    private State state = new State();

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
