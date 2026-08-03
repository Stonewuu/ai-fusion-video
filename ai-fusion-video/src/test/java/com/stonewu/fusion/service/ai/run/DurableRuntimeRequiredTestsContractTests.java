package com.stonewu.fusion.service.ai.run;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestTemplate;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.platform.commons.support.AnnotationSupport;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

class DurableRuntimeRequiredTestsContractTests {

    private static final Set<String> CONDITIONAL_ANNOTATIONS = Set.of(
            "Disabled", "DisabledIf", "EnabledIf",
            "DisabledIfEnvironmentVariable", "EnabledIfEnvironmentVariable",
            "DisabledIfSystemProperty", "EnabledIfSystemProperty",
            "DisabledOnOs", "EnabledOnOs", "DisabledOnJre", "EnabledOnJre",
            "DisabledForJreRange", "EnabledForJreRange");

    @ParameterizedTest(name = "required durable runtime IT: {0}")
    @ValueSource(strings = {
            "AgentPersistenceMigrationIT",
            "AgentMessageAllocatorIT",
            "AgentRunStartIT",
            "AgentJournalTerminalIT",
            "AgentModelCallUsageIT",
            "AgentOutboxMultiInstanceIT",
            "AgentProjectionRecoveryIT",
            "PlatformSubAgentRunServiceIT",
            "AgentWaitingStateIT",
            "AgentFencingCancellationIT",
            "AgentOwnedJournalTakeoverIT",
            "AgentReplayLiveIT",
            "AgentIntegrationProfileSentinelIT",
            "AgentDurableRuntimeMultiInstanceIT"
    })
    void requiredIntegrationTestExistsAndCannotBeSkipped(String simpleName) {
        String qualifiedName = "com.stonewu.fusion.integration." + simpleName;
        Class<?> type = assertDoesNotThrow(() -> Class.forName(
                qualifiedName,
                false,
                getClass().getClassLoader()));

        assertThat(annotationNames(type.getAnnotations()))
                .doesNotContainAnyElementsOf(CONDITIONAL_ANNOTATIONS);
        List<Method> tests = Arrays.stream(type.getDeclaredMethods())
                .filter(method -> AnnotationSupport.isAnnotated(method, Test.class)
                        || AnnotationSupport.isAnnotated(method, TestTemplate.class))
                .toList();
        assertThat(tests).isNotEmpty();
        assertThat(tests).allSatisfy(method ->
                assertThat(annotationNames(method.getAnnotations()))
                        .doesNotContainAnyElementsOf(CONDITIONAL_ANNOTATIONS));

        Path source = Path.of(
                "src", "test", "java", "com", "stonewu", "fusion",
                "integration", simpleName + ".java");
        String sourceText = assertDoesNotThrow(() -> Files.readString(source));
        assertThat(sourceText).doesNotContain(
                "Assumptions.",
                "assumeTrue(",
                "assumeFalse(",
                "TestAbortedException",
                "disabledWithoutDocker = true",
                "disabledWithoutDocker=true");
    }

    private List<String> annotationNames(java.lang.annotation.Annotation[] annotations) {
        return Arrays.stream(annotations)
                .map(annotation -> annotation.annotationType().getSimpleName())
                .toList();
    }
}
