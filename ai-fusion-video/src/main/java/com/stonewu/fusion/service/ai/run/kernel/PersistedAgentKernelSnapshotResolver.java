package com.stonewu.fusion.service.ai.run.kernel;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class PersistedAgentKernelSnapshotResolver implements AgentKernelSnapshotResolver {

    private final ObjectMapper objectMapper;
    private final AgentKernelSnapshotBuilder snapshotBuilder;

    public PersistedAgentKernelSnapshotResolver() {
        this(new ObjectMapper());
    }

    public PersistedAgentKernelSnapshotResolver(ObjectMapper objectMapper) {
        this(objectMapper, new CanonicalAgentKernelSnapshotBuilder(objectMapper));
    }

    public PersistedAgentKernelSnapshotResolver(
            ObjectMapper objectMapper,
            AgentKernelSnapshotBuilder snapshotBuilder) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null")
                .copy();
        this.objectMapper.enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION);
        this.objectMapper.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        this.snapshotBuilder = Objects.requireNonNull(
                snapshotBuilder, "snapshotBuilder must not be null");
    }

    @Override
    public AgentKernelSnapshot resolve(
            String persistedCanonicalJson,
            String persistedFingerprint,
            long currentModelConfigVersion,
            List<ToolManifestSnapshot> currentTools) {
        try {
            Objects.requireNonNull(persistedCanonicalJson, "persistedCanonicalJson must not be null");
            Objects.requireNonNull(persistedFingerprint, "persistedFingerprint must not be null");
            Objects.requireNonNull(currentTools, "currentTools must not be null");

            byte[] persistedBytes = persistedCanonicalJson.getBytes(StandardCharsets.UTF_8);
            byte[] expectedFingerprint = AgentKernelSnapshot.sha256(persistedBytes);
            byte[] suppliedFingerprint = parseFingerprint(persistedFingerprint);
            if (!MessageDigest.isEqual(expectedFingerprint, suppliedFingerprint)) {
                throw unavailable("Persisted kernel fingerprint does not match its payload");
            }

            JsonNode root = objectMapper.readTree(persistedCanonicalJson);
            if (root == null || !root.isObject()) {
                throw unavailable("Persisted kernel snapshot is not a JSON object");
            }
            JsonNode schemaVersion = root.get("schemaVersion");
            if (schemaVersion == null
                    || !schemaVersion.isIntegralNumber()
                    || schemaVersion.intValue()
                        != AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION) {
                throw unavailable("Persisted kernel snapshot schema is unsupported");
            }

            AgentKernelSnapshotPayload payload = objectMapper.treeToValue(
                    root, AgentKernelSnapshotPayload.class);
            AgentKernelSnapshot resolved = snapshotBuilder.build(payload);
            if (!MessageDigest.isEqual(resolved.canonicalBytes(), persistedBytes)) {
                throw unavailable("Persisted kernel snapshot is not canonical JSON");
            }
            validateAvailability(payload, currentModelConfigVersion, currentTools);
            return resolved;
        } catch (RunConfigUnavailableException unavailable) {
            throw unavailable;
        } catch (Exception invalidSnapshot) {
            throw unavailable("Persisted kernel configuration is unavailable");
        }
    }

    private void validateAvailability(
            AgentKernelSnapshotPayload payload,
            long currentModelConfigVersion,
            List<ToolManifestSnapshot> currentTools) {
        if (payload.modelConfigVersion() != currentModelConfigVersion) {
            throw unavailable("Persisted model configuration version is unavailable");
        }

        Map<String, ToolManifestSnapshot> availableByName = indexTools(currentTools);
        if (availableByName.size() != payload.tools().size()) {
            throw unavailable("Persisted tool set is unavailable");
        }
        for (ToolManifestSnapshot persisted : payload.tools()) {
            ToolManifestSnapshot available = availableByName.get(persisted.name());
            if (available == null
                    || !persisted.implementationVersion()
                        .equals(available.implementationVersion())
                    || !persisted.schemaSha256().equals(available.schemaSha256())
                    || persisted.readOnly() != available.readOnly()
                    || persisted.concurrencySafe() != available.concurrencySafe()) {
                throw unavailable("Persisted tool implementation is unavailable");
            }
        }
    }

    private Map<String, ToolManifestSnapshot> indexTools(List<ToolManifestSnapshot> tools) {
        Map<String, ToolManifestSnapshot> byName = new HashMap<>();
        for (ToolManifestSnapshot tool : tools) {
            Objects.requireNonNull(tool, "currentTools entry must not be null");
            if (byName.putIfAbsent(tool.name(), tool) != null) {
                throw unavailable("Current tool manifest contains duplicate names");
            }
        }
        return byName;
    }

    private byte[] parseFingerprint(String fingerprint) {
        String normalized = fingerprint.trim();
        if (!normalized.matches("[0-9a-fA-F]{64}")) {
            throw unavailable("Persisted kernel fingerprint is invalid");
        }
        return HexFormat.of().parseHex(normalized);
    }

    private static RunConfigUnavailableException unavailable(String message) {
        return new RunConfigUnavailableException(message);
    }
}
