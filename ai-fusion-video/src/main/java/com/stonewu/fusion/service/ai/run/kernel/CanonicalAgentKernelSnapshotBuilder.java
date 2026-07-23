package com.stonewu.fusion.service.ai.run.kernel;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

@Component
public final class CanonicalAgentKernelSnapshotBuilder implements AgentKernelSnapshotBuilder {

    private static final String APPLICATION_VERSION = "ai-fusion-video-p2";

    private static final Set<String> SECRET_FIELD_NAMES = Set.of(
            "authorization",
            "apikey",
            "apisecret",
            "appsecret",
            "clientsecret",
            "secret",
            "password",
            "proxypassword",
            "privatekey",
            "credential",
            "credentials",
            "accesstoken",
            "refreshtoken",
            "idtoken",
            "signingkey");

    private static final Pattern AUTHORIZATION_VALUE = Pattern.compile(
            "(?i)(?:authorization\\s*[:=]\\s*)?(?:bearer|basic)\\s+[A-Za-z0-9._~+/=-]{8,}");
    private static final Pattern DATA_URI = Pattern.compile(
            "(?i)data:[^\\s,]+(?:;[^\\s,]+)*,");
    private static final Pattern SIGNED_QUERY = Pattern.compile(
            "(?i)[?&](?:x-amz-signature|x-goog-signature|signature|sig|token|access_token|"
                    + "expires|ossaccesskeyid|googleaccessid)=");
    private static final Pattern BASE64 = Pattern.compile("[A-Za-z0-9+/]+={0,2}");
    private static final Pattern SHA256_HEX = Pattern.compile("[0-9a-fA-F]{64}");

    private final ObjectMapper objectMapper;
    private final AiModelMetadataResolver modelMetadataResolver;

    public CanonicalAgentKernelSnapshotBuilder() {
        this(new ObjectMapper(), null);
    }

    public CanonicalAgentKernelSnapshotBuilder(ObjectMapper objectMapper) {
        this(objectMapper, null);
    }

    @Autowired
    public CanonicalAgentKernelSnapshotBuilder(ObjectMapper objectMapper,
                                               AiModelMetadataResolver modelMetadataResolver) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null")
                .copy();
        this.modelMetadataResolver = modelMetadataResolver;
        this.objectMapper.enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION);
        this.objectMapper.enable(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY);
        this.objectMapper.enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);
        this.objectMapper.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    }

    @Override
    public AgentKernelSnapshot build(AgentKernelSnapshotPayload payload) {
        Objects.requireNonNull(payload, "payload must not be null");
        try {
            validateSafeText(payload.agentDefinitionStableKey());
            validateSafeText(payload.agentName());
            validateSafeText(payload.description());
            validateSafeText(payload.systemPrompt());
            payload.promptVariables().forEach((name, value) -> {
                validateSafeText(name);
                validateSafeText(value);
            });
            validateSafeText(payload.modelConfigId());
            validateSafeText(payload.provider());
            validateSafeText(payload.modelCode());
            validateSafeText(payload.applicationVersion());

            JsonNode sortedOptions = normalizeNode(payload.modelOptions());
            List<ToolManifestSnapshot> sortedTools = payload.tools().stream()
                    .sorted(Comparator.comparing(ToolManifestSnapshot::name))
                    .toList();
            AgentKernelSnapshotPayload normalized = new AgentKernelSnapshotPayload(
                    payload.schemaVersion(),
                    payload.agentDefinitionStableKey(),
                    payload.agentName(),
                    payload.description(),
                    payload.systemPrompt(),
                    payload.promptVariables(),
                    payload.maxIters(),
                    payload.modelConfigId(),
                    payload.modelConfigVersion(),
                    payload.provider(),
                    payload.modelCode(),
                    sortedOptions,
                    sortedTools,
                    payload.applicationVersion());
            byte[] canonicalBytes = objectMapper.writeValueAsBytes(normalized);
            String canonicalJson = new String(canonicalBytes, StandardCharsets.UTF_8);
            return new AgentKernelSnapshot(
                    normalized,
                    canonicalJson,
                    AgentKernelSnapshot.fingerprint(canonicalBytes));
        } catch (RunConfigUnavailableException unavailable) {
            throw unavailable;
        } catch (JsonProcessingException | IllegalArgumentException invalidPayload) {
            throw unavailable("Kernel snapshot payload is unavailable");
        }
    }

    @Override
    public AgentKernelSnapshot build(AgentKernelSpec spec) {
        AgentKernelSpec safeSpec = Objects.requireNonNull(spec, "spec must not be null");
        AiModel model = safeSpec.model();
        List<ToolManifestSnapshot> tools = safeSpec.toolManifest().stream()
                .map(tool -> new ToolManifestSnapshot(
                        tool.toolName(),
                        tool.schemaSha256(),
                        tool.readOnly(),
                        tool.concurrencySafe(),
                        safeSpec.toolWhitelistVersion()))
                .toList();
        return build(safeSpec.snapshotPayload(
                requireModelId(model),
                modelConfigVersion(safeSpec.key().modelConfigFingerprint()),
                provider(model),
                parseModelOptions(model.getConfig()),
                tools,
                APPLICATION_VERSION));
    }

    public static long modelConfigVersion(String modelConfigFingerprint) {
        String fingerprint = Objects.requireNonNull(
                modelConfigFingerprint, "modelConfigFingerprint must not be null");
        if (!fingerprint.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalArgumentException("modelConfigFingerprint must be SHA-256 hex");
        }
        long value = Long.parseUnsignedLong(fingerprint.substring(0, 15), 16);
        return value == 0L ? 1L : value;
    }

    private JsonNode parseModelOptions(String config) {
        if (config == null || config.isBlank()) {
            return objectMapper.createObjectNode();
        }
        try {
            JsonNode parsed = objectMapper.readTree(config);
            if (parsed == null || !parsed.isObject()) {
                throw new IllegalArgumentException("model config must be a JSON object");
            }
            return parsed;
        } catch (JsonProcessingException invalid) {
            throw new IllegalArgumentException("model config is invalid JSON", invalid);
        }
    }

    private String requireModelId(AiModel model) {
        if (model.getId() == null || model.getId() <= 0) {
            throw new IllegalArgumentException("model.id must be positive");
        }
        return String.valueOf(model.getId());
    }

    private String provider(AiModel model) {
        String protocol = modelMetadataResolver != null
                ? modelMetadataResolver.resolve(model).modelProtocol()
                : model.getModelProtocol();
        if (protocol == null || protocol.isBlank()) {
            throw new IllegalArgumentException("model provider must not be blank");
        }
        return protocol.trim();
    }

    private JsonNode normalizeNode(JsonNode node) {
        if (node.isObject()) {
            Map<String, JsonNode> sorted = new TreeMap<>();
            node.fields().forEachRemaining(entry -> sorted.put(entry.getKey(), entry.getValue()));
            ObjectNode result = JsonNodeFactory.instance.objectNode();
            for (Map.Entry<String, JsonNode> entry : sorted.entrySet()) {
                rejectSecretField(entry.getKey());
                result.set(entry.getKey(), normalizeNode(entry.getValue()));
            }
            return result;
        }
        if (node.isArray()) {
            ArrayNode result = JsonNodeFactory.instance.arrayNode();
            for (JsonNode item : node) {
                result.add(normalizeNode(item));
            }
            return result;
        }
        if (node.isTextual()) {
            validateSafeText(node.textValue());
        } else if (node.isBinary() || node.isPojo()) {
            throw unavailable("Kernel model options contain unsupported binary content");
        } else if (node.isFloatingPointNumber() && !Double.isFinite(node.doubleValue())) {
            throw unavailable("Kernel model options contain a non-finite number");
        }
        return node.deepCopy();
    }

    private void rejectSecretField(String fieldName) {
        String normalized = fieldName.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
        if (SECRET_FIELD_NAMES.contains(normalized)
                || normalized.endsWith("password")
                || normalized.endsWith("secret")
                || normalized.endsWith("privatekey")
                || normalized.endsWith("apikey")
                || normalized.endsWith("accesstoken")
                || normalized.endsWith("refreshtoken")) {
            throw unavailable("Kernel model options contain a forbidden secret field");
        }
    }

    private void validateSafeText(String value) {
        if (AUTHORIZATION_VALUE.matcher(value).find()
                || DATA_URI.matcher(value).find()
                || containsSignedQuery(value)
                || isRawBase64(value.trim())) {
            throw unavailable("Kernel snapshot contains forbidden sensitive content");
        }
    }

    private boolean containsSignedQuery(String value) {
        if (SIGNED_QUERY.matcher(value).find()) {
            return true;
        }
        try {
            String decoded = URLDecoder.decode(value, StandardCharsets.UTF_8);
            return !decoded.equals(value) && SIGNED_QUERY.matcher(decoded).find();
        } catch (IllegalArgumentException invalidEncoding) {
            return false;
        }
    }

    private boolean isRawBase64(String value) {
        if (value.length() < 8
                || value.length() % 4 == 1
                || SHA256_HEX.matcher(value).matches()
                || !BASE64.matcher(value).matches()
                || !(value.endsWith("=")
                    || value.indexOf('+') >= 0
                    || value.indexOf('/') >= 0
                    || value.length() >= 64)) {
            return false;
        }
        try {
            return Base64.getDecoder().decode(value).length >= 4;
        } catch (IllegalArgumentException notBase64) {
            return false;
        }
    }

    private static RunConfigUnavailableException unavailable(String message) {
        return new RunConfigUnavailableException(message);
    }
}
