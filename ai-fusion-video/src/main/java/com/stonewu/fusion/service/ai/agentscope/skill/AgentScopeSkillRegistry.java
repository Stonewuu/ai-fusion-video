package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import io.agentscope.core.skill.AgentSkill;
import io.agentscope.core.skill.repository.AgentSkillRepository;
import io.agentscope.core.skill.repository.ClasspathSkillRepository;
import io.agentscope.core.skill.repository.FileSystemSkillRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.stereotype.Component;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Builds the explicitly configured, application-owned AgentScope skill repositories. */
@Component
public final class AgentScopeSkillRegistry implements DisposableBean {

    private static final Logger log = LoggerFactory.getLogger(AgentScopeSkillRegistry.class);
    private static final String CLASSPATH_PREFIX = "classpath:";
    private static final String FILE_PREFIX = "file:";

    private final List<AgentSkillRepository> repositories;
    private final Map<String, String> displayNames;

    public AgentScopeSkillRegistry(AgentScopeV2Properties properties) {
        AgentScopeV2Properties.Skills config = properties.getSkills();
        this.repositories = config.isEnabled()
                ? loadRepositories(config)
                : List.of();
        this.displayNames = Map.copyOf(config.getDisplayNames());
        log.info("AgentScope skills initialized [enabled={}, repositories={}]",
                enabled(), repositories.size());
    }

    public boolean enabled() {
        return !repositories.isEmpty();
    }

    public List<AgentSkillRepository> repositories() {
        return repositories;
    }

    /** Returns the effective low-to-high precedence catalog exposed by AgentScope. */
    public List<SkillReference> catalog() {
        return skills().stream()
                .map(skill -> {
                    String displayName = requireText(
                            displayNames.get(skill.getName()),
                            "display name for skill " + skill.getName());
                    return new SkillReference(
                            skill.getSkillId(),
                            skill.getName(),
                            displayName,
                            skill.getDescription(),
                            skill.getSource());
                })
                .toList();
    }

    /** Returns the effective Skill documents using the same precedence as the Harness runtime. */
    public List<AgentSkill> skills() {
        Map<String, AgentSkill> merged = new LinkedHashMap<>();
        for (AgentSkillRepository repository : repositories) {
            try {
                for (AgentSkill skill : repository.getAllSkills()) {
                    if (skill == null) {
                        continue;
                    }
                    merged.put(skill.getName(), skill);
                }
            } catch (RuntimeException failure) {
                log.warn("Failed to read AgentScope skill catalog from '{}': {}",
                        repository.getSource(), failure.getMessage());
            }
        }
        return merged.values().stream()
                .sorted(Comparator.comparing(AgentSkill::getName))
                .toList();
    }

    @Override
    public void destroy() {
        RuntimeException failure = null;
        for (AgentSkillRepository repository : repositories) {
            if (!(repository instanceof AutoCloseable closeable)) {
                continue;
            }
            try {
                closeable.close();
            } catch (Exception closeFailure) {
                if (failure == null) {
                    failure = new IllegalStateException(
                            "Failed to close AgentScope skill repositories", closeFailure);
                } else {
                    failure.addSuppressed(closeFailure);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private List<AgentSkillRepository> loadRepositories(
            AgentScopeV2Properties.Skills config) {
        List<AgentSkillRepository> loaded = new ArrayList<>();
        for (Map.Entry<String, AgentScopeV2Properties.SkillRepository> entry
                : config.getRepositories().entrySet()) {
            String name = requireText(entry.getKey(), "skill repository name");
            AgentScopeV2Properties.SkillRepository repository = entry.getValue();
            if (repository == null || !repository.isEnabled()) {
                continue;
            }
            try {
                loaded.add(loadRepository(name, repository));
            } catch (Exception failure) {
                if (config.isFailFast()) {
                    closeQuietly(loaded);
                    throw new IllegalStateException(
                            "Failed to initialize AgentScope skill repository: " + name,
                            failure);
                }
                log.warn("Skipping unavailable AgentScope skill repository '{}': {}",
                        name, failure.getMessage());
            }
        }
        return List.copyOf(loaded);
    }

    private AgentSkillRepository loadRepository(
            String name,
            AgentScopeV2Properties.SkillRepository config) throws Exception {
        String location = requireText(config.getLocation(),
                "skill repository " + name + " location");
        if (location.toLowerCase(Locale.ROOT).startsWith(CLASSPATH_PREFIX)) {
            String resourcePath = requireText(
                    location.substring(CLASSPATH_PREFIX.length()),
                    "classpath skill repository path");
            return new ClasspathSkillRepository(stripLeadingSlash(resourcePath), name);
        }
        String pathValue = location.toLowerCase(Locale.ROOT).startsWith(FILE_PREFIX)
                ? location.substring(FILE_PREFIX.length())
                : location;
        Path path = Path.of(requireText(pathValue, "filesystem skill repository path"));
        return new FileSystemSkillRepository(path, false, name, config.isLazy());
    }

    private void closeQuietly(List<AgentSkillRepository> loaded) {
        for (AgentSkillRepository repository : loaded) {
            if (repository instanceof AutoCloseable closeable) {
                try {
                    closeable.close();
                } catch (Exception ignored) {
                    // Preserve the initialization failure as the primary cause.
                }
            }
        }
    }

    private String stripLeadingSlash(String value) {
        String normalized = value.replace('\\', '/');
        return normalized.startsWith("/") ? normalized.substring(1) : normalized;
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value.trim();
    }

    public record SkillReference(
            String id,
            String name,
            String displayName,
            String description,
            String source) {
    }
}
