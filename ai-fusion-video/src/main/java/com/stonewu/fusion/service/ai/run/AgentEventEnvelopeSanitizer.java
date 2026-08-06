package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import org.springframework.stereotype.Component;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

@Component
public final class AgentEventEnvelopeSanitizer {

    static final String SECRET_REDACTED = "[SECRET_REDACTED]";
    static final String BINARY_REDACTED = "[BINARY_REDACTED]";
    static final String SIGNED_URL_REDACTED = "[SIGNED_URL_REDACTED]";
    static final String FILE_PATH_REDACTED = "[FILE_PATH_REDACTED]";

    private static final Set<String> SECRET_FIELDS = Set.of(
            "authorization",
            "credential",
            "credentials",
            "apikey",
            "apisecret",
            "password",
            "proxypassword",
            "secret",
            "clientsecret",
            "appsecret",
            "accesstoken",
            "refreshtoken",
            "privatekey");

    private static final Pattern AUTHORIZATION_VALUE = Pattern.compile(
            "(?i)(?:authorization\\s*[:=]\\s*)?(?:bearer|basic)\\s+[A-Za-z0-9._~+/=-]{8,}");
    private static final Pattern DATA_URI = Pattern.compile(
            "(?i)data:[^\\s,]+(?:;[^\\s,]+)*,");
    private static final Pattern SIGNED_QUERY = Pattern.compile(
            "(?i)[?&](?:x-amz-signature|x-goog-signature|signature|sig|token|access_token|"
                    + "expires|ossaccesskeyid|googleaccessid)=");
    private static final Pattern ABSOLUTE_FILE_PATH = Pattern.compile(
            "(?i)(?:^|\\s)(?:file:/{1,3}|[a-z]:[\\\\/]|\\\\\\\\|/"
                    + "(?:app|data|dev|etc|home|mnt|opt|private|proc|root|run|srv|sys|tmp|usr|"
                    + "users|var|volumes|workspace)"
                    + "(?:[\\\\/]|$))[^\\s]*");
    private static final Pattern PATH_TRAVERSAL = Pattern.compile(
            "(?:^|[\\\\/])\\.\\.(?:[\\\\/]|$)");
    private static final Pattern BASE64 = Pattern.compile("[A-Za-z0-9+/]+={0,2}");
    private static final Pattern SHA256_HEX = Pattern.compile("[0-9a-fA-F]{64}");

    private final ObjectMapper objectMapper;

    public AgentEventEnvelopeSanitizer(ObjectMapper objectMapper) {
        this.objectMapper = Objects.requireNonNull(
                objectMapper, "objectMapper must not be null");
    }

    public JsonNode sanitize(JsonNode payload) {
        return sanitizeNode(Objects.requireNonNull(payload, "payload must not be null"));
    }

    private JsonNode sanitizeNode(JsonNode node) {
        if (node.isObject()) {
            ObjectNode sanitized = JsonNodeFactory.instance.objectNode();
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                sanitized.set(
                        field.getKey(),
                        isSecretField(field.getKey())
                                ? TextNode.valueOf(SECRET_REDACTED)
                                : sanitizeNode(field.getValue()));
            }
            return sanitized;
        }
        if (node.isArray()) {
            ArrayNode sanitized = JsonNodeFactory.instance.arrayNode();
            for (JsonNode item : node) {
                sanitized.add(sanitizeNode(item));
            }
            return sanitized;
        }
        if (node.isBinary() || node.isPojo()) {
            return TextNode.valueOf(BINARY_REDACTED);
        }
        if (!node.isTextual()) {
            return node.deepCopy();
        }

        String value = node.textValue();
        JsonNode structuredJson = parseStructuredJson(value);
        if (structuredJson != null) {
            return TextNode.valueOf(sanitizeNode(structuredJson).toString());
        }
        if (AUTHORIZATION_VALUE.matcher(value).find()) {
            return TextNode.valueOf(SECRET_REDACTED);
        }
        if (DATA_URI.matcher(value).find() || isRawBase64(value.trim())) {
            return TextNode.valueOf(BINARY_REDACTED);
        }
        if (containsSignedQuery(value)) {
            return TextNode.valueOf(SIGNED_URL_REDACTED);
        }
        if (ABSOLUTE_FILE_PATH.matcher(value).find() || PATH_TRAVERSAL.matcher(value).find()) {
            return TextNode.valueOf(FILE_PATH_REDACTED);
        }
        return TextNode.valueOf(value);
    }

    private JsonNode parseStructuredJson(String value) {
        String trimmed = value.trim();
        boolean object = trimmed.startsWith("{") && trimmed.endsWith("}");
        boolean array = trimmed.startsWith("[") && trimmed.endsWith("]");
        if (!object && !array) {
            return null;
        }
        try {
            JsonNode parsed = objectMapper.readTree(trimmed);
            return parsed != null && (parsed.isObject() || parsed.isArray())
                    ? parsed
                    : null;
        } catch (JsonProcessingException invalidJson) {
            return null;
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

    private boolean isSecretField(String name) {
        String normalized = name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
        return SECRET_FIELDS.contains(normalized)
                || normalized.endsWith("password")
                || normalized.endsWith("secret")
                || normalized.endsWith("apikey")
                || normalized.endsWith("accesstoken")
                || normalized.endsWith("refreshtoken")
                || normalized.endsWith("credential");
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
}
