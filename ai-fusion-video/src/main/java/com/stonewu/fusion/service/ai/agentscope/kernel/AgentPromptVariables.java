package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.controller.ai.vo.AiReferenceVO;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Immutable prompt variables that remain part of a durable Harness kernel definition. */
public final class AgentPromptVariables {

    private static final Pattern VARIABLE_NAME = Pattern.compile("[A-Za-z][A-Za-z0-9]*");
    private static final Pattern TEMPLATE_VARIABLE = Pattern.compile("\\{([A-Za-z][A-Za-z0-9]*)}");

    private AgentPromptVariables() {
    }

    public static Map<String, String> fromRequest(AiChatReqVO request) {
        AiChatReqVO safeRequest = Objects.requireNonNull(request, "request must not be null");
        Map<String, String> variables = new LinkedHashMap<>();
        if (safeRequest.getContext() != null) {
            safeRequest.getContext().forEach((name, value) -> putScalar(variables, name, value));
        }
        if (safeRequest.getAutoReferences() != null) {
            for (AiReferenceVO reference : safeRequest.getAutoReferences()) {
                if (reference == null || reference.getId() == null) {
                    continue;
                }
                String type = normalize(reference.getType());
                if (type != null) {
                    putScalar(variables, type + "Id", reference.getId());
                }
            }
        }
        putScalar(variables, "projectId", safeRequest.getProjectId());
        return immutable(variables);
    }

    public static Map<String, String> forChild(
            Map<String, String> inherited,
            ProjectContext project,
            Map<String, Object> input) {
        Map<String, String> variables = new LinkedHashMap<>(immutable(inherited));
        if (project != null) {
            putScalar(variables, "projectId", project.projectId());
        }
        if (input != null) {
            input.forEach((name, value) -> putScalar(variables, name, value));
        }
        return immutable(variables);
    }

    public static Map<String, String> immutable(Map<String, String> variables) {
        Objects.requireNonNull(variables, "promptVariables must not be null");
        Map<String, String> sorted = new TreeMap<>();
        variables.forEach((name, value) -> {
            String safeName = requireName(name);
            String safeValue = Objects.requireNonNull(
                    value, "prompt variable value must not be null: " + safeName);
            sorted.put(safeName, safeValue);
        });
        return Collections.unmodifiableMap(sorted);
    }

    public static String render(String template, Map<String, String> variables) {
        String rendered = requireText(template, "template");
        for (Map.Entry<String, String> variable : immutable(variables).entrySet()) {
            rendered = rendered.replace(
                    "{" + variable.getKey() + "}", variable.getValue());
        }
        Matcher unresolved = TEMPLATE_VARIABLE.matcher(rendered);
        if (unresolved.find()) {
            throw new IllegalArgumentException(
                    "Unresolved Agent prompt variable: " + unresolved.group(1));
        }
        return rendered;
    }

    public static String canonical(Map<String, String> variables) {
        StringBuilder canonical = new StringBuilder();
        immutable(variables).forEach((name, value) -> canonical
                .append(name.length()).append(':').append(name).append('\n')
                .append(value.length()).append(':').append(value).append('\n'));
        return canonical.toString();
    }

    private static void putScalar(Map<String, String> target, String name, Object value) {
        if (value == null) {
            return;
        }
        if (!(value instanceof String || value instanceof Number || value instanceof Boolean)) {
            return;
        }
        target.put(requireName(name), String.valueOf(value));
    }

    private static String requireName(String name) {
        String normalized = normalize(name);
        if (normalized == null || !VARIABLE_NAME.matcher(normalized).matches()) {
            throw new IllegalArgumentException("Invalid Agent prompt variable name: " + name);
        }
        return normalized;
    }

    private static String requireText(String value, String field) {
        String normalized = normalize(value);
        if (normalized == null) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return normalized;
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
