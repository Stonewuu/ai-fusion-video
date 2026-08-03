package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import org.springframework.stereotype.Component;

import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.util.UUID;

@Component
public final class AgentRuntimeInstanceIdentity {

    private final String value;

    public AgentRuntimeInstanceIdentity(AgentScopeV2Properties properties) {
        String configured = properties.getExecution().getInstanceId();
        this.value = configured != null ? configured : generatedIdentity();
    }

    public String value() {
        return value;
    }

    private String generatedIdentity() {
        String host;
        try {
            host = InetAddress.getLocalHost().getHostName();
        } catch (Exception unavailable) {
            host = "unknown-host";
        }
        String process = ManagementFactory.getRuntimeMXBean().getName();
        String suffix = UUID.randomUUID().toString().substring(0, 8);
        String generated = host + ':' + process + ':' + suffix;
        return generated.length() <= 128 ? generated : generated.substring(0, 128);
    }
}
