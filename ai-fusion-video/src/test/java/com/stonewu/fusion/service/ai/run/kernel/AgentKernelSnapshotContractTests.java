package com.stonewu.fusion.service.ai.run.kernel;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolManifest;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowableOfType;

class AgentKernelSnapshotContractTests {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final AgentKernelSnapshotBuilder builder =
            new CanonicalAgentKernelSnapshotBuilder(objectMapper);
    private final AgentKernelSnapshotResolver resolver =
            new PersistedAgentKernelSnapshotResolver(objectMapper, builder);

    @Test
    void canonicalizesModelOptionsRecursivelyAndSortsToolsByName() throws Exception {
        AgentKernelSnapshot first = builder.build(payload(
                objectMapper.readTree("""
                        {"z":{"b":2,"a":1},"a":[{"d":4,"c":3}]}
                        """),
                List.of(tool("zeta", "zeta-v1", "zeta-schema"),
                        tool("alpha", "alpha-v1", "alpha-schema"))));
        AgentKernelSnapshot second = builder.build(payload(
                objectMapper.readTree("""
                        {"a":[{"c":3,"d":4}],"z":{"a":1,"b":2}}
                        """),
                List.of(tool("alpha", "alpha-v1", "alpha-schema"),
                        tool("zeta", "zeta-v1", "zeta-schema"))));

        assertThat(first.snapshotJson()).isEqualTo(second.snapshotJson());
        assertThat(first.fingerprint()).isEqualTo(second.fingerprint());

        JsonNode canonical = objectMapper.readTree(first.snapshotJson());
        assertThat(fieldNames(canonical.path("modelOptions")))
                .containsExactly("a", "z");
        assertThat(fieldNames(canonical.path("modelOptions").path("z")))
                .containsExactly("a", "b");
        assertThat(fieldNames(canonical.path("modelOptions").path("a").get(0)))
                .containsExactly("c", "d");
        assertThat(canonical.path("tools").findValuesAsText("name"))
                .containsExactly("alpha", "zeta");
    }

    @Test
    void fingerprintsTheExactCanonicalUtf8BytesWithSha256() throws Exception {
        AgentKernelSnapshot snapshot = builder.build(payload(
                objectMapper.readTree("{\"temperature\":0.2}"),
                List.of(tool("alpha", "alpha-v1", "alpha-schema"))));

        byte[] expected = MessageDigest.getInstance("SHA-256")
                .digest(snapshot.snapshotJson().getBytes(StandardCharsets.UTF_8));

        assertThat(snapshot.fingerprint()).isEqualTo(HexFormat.of().formatHex(expected));
    }

    @Test
    void payloadDefensivelyCopiesMutableModelOptions() throws Exception {
        ObjectNode options = (ObjectNode) objectMapper.readTree("{\"nested\":{\"value\":1}}");
        AgentKernelSnapshotPayload payload = payload(options, List.of());
        options.put("lateSecret", "must-not-appear");
        ((ObjectNode) payload.modelOptions()).put("otherMutation", true);

        AgentKernelSnapshot snapshot = builder.build(payload);

        assertThat(snapshot.snapshotJson())
                .doesNotContain("lateSecret")
                .doesNotContain("otherMutation");
    }

    @Test
    void rejectsSecretsAuthorizationProxyPasswordsBase64DataUrisAndSignedQueries()
            throws Exception {
        List<String> forbiddenOptions = List.of(
                "{\"apiKey\":\"not-persistable\"}",
                "{\"header\":\"Authorization: Bearer abcdefgh1234\"}",
                "{\"proxy_password\":\"not-persistable\"}",
                "{\"blob\":\"aGVsbG8=\"}",
                "{\"image\":\"data:image/png;base64,aGVsbG8=\"}",
                "{\"inline\":\"data:text/plain,hello\"}",
                "{\"url\":\"https://media.test/a?X-Amz-Signature=abcdef\"}");

        for (String optionsJson : forbiddenOptions) {
            RunConfigUnavailableException failure = catchThrowableOfType(
                    () -> builder.build(payload(objectMapper.readTree(optionsJson), List.of())),
                    RunConfigUnavailableException.class);
            assertRunConfigUnavailable(failure);
        }
    }

    @Test
    void resolvesOnlyExactModelAndToolVersions() throws Exception {
        List<ToolManifestSnapshot> tools = List.of(
                tool("alpha", "alpha-v1", "alpha-schema"),
                tool("zeta", "zeta-v1", "zeta-schema"));
        AgentKernelSnapshot snapshot = builder.build(payload(
                objectMapper.readTree("{\"temperature\":0.2}"), tools));

        AgentKernelSnapshot resolved = resolver.resolve(
                snapshot.snapshotJson(), snapshot.fingerprint(), 7L, tools);

        assertThat(resolved).isEqualTo(snapshot);

        assertUnavailable(() -> resolver.resolve(
                snapshot.snapshotJson(), snapshot.fingerprint(), 8L, tools));
        String missingModelVersion = snapshot.snapshotJson()
                .replace("\"modelConfigVersion\":7", "\"modelConfigVersion\":0");
        assertThat(missingModelVersion).contains("\"modelConfigVersion\":0");
        assertUnavailable(() -> resolver.resolve(
                missingModelVersion, fingerprint(missingModelVersion), 0L, tools));
        assertUnavailable(() -> resolver.resolve(
                snapshot.snapshotJson(), snapshot.fingerprint(), 7L,
                List.of(tool("alpha", "alpha-v2", "alpha-schema"), tools.get(1))));
        assertUnavailable(() -> resolver.resolve(
                snapshot.snapshotJson(), snapshot.fingerprint(), 7L,
                List.of(tool("alpha", "alpha-v1", "changed-schema"), tools.get(1))));
    }

    @Test
    void rejectsFingerprintSchemaAndNonCanonicalPayloadsAsUnavailable() throws Exception {
        AgentKernelSnapshot snapshot = builder.build(payload(
                objectMapper.readTree("{\"temperature\":0.2}"), List.of()));

        String wrongFingerprint = "0".repeat(64);
        assertUnavailable(() -> resolver.resolve(
                snapshot.snapshotJson(), wrongFingerprint, 7L, List.of()));

        String unsupportedSchema = snapshot.snapshotJson()
                .replace("\"schemaVersion\":2", "\"schemaVersion\":99");
        assertUnavailable(() -> resolver.resolve(
                unsupportedSchema,
                fingerprint(unsupportedSchema),
                7L,
                List.of()));

        String nonCanonical = objectMapper.writerWithDefaultPrettyPrinter()
                .writeValueAsString(objectMapper.readTree(snapshot.snapshotJson()));
        assertUnavailable(() -> resolver.resolve(
                nonCanonical,
                fingerprint(nonCanonical),
                7L,
                List.of()));
    }

    private AgentKernelSnapshotPayload payload(
            JsonNode modelOptions,
            List<ToolManifestSnapshot> tools) {
        return new AgentKernelSnapshotPayload(
                AgentKernelSnapshotPayload.CURRENT_SCHEMA_VERSION,
                "storyboard_creator",
                "Storyboard Creator",
                "Creates storyboard scenes",
                "Create a consistent storyboard.",
                20,
                "model-config-42",
                7L,
                "dashscope",
                "qwen-plus",
                modelOptions,
                tools,
                "1.0.0");
    }

    private ToolManifestSnapshot tool(String name, String version, String schema) {
        return new ToolManifestSnapshot(
                name,
                AgentKernelToolManifest.schemaSha256(schema),
                false,
                false,
                version);
    }

    private List<String> fieldNames(JsonNode object) {
        List<String> names = new ArrayList<>();
        object.fieldNames().forEachRemaining(names::add);
        return names;
    }

    private String fingerprint(String json) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(json.getBytes(StandardCharsets.UTF_8)));
    }

    private void assertUnavailable(ThrowingCall call) {
        RunConfigUnavailableException failure = catchThrowableOfType(
                call::run, RunConfigUnavailableException.class);
        assertRunConfigUnavailable(failure);
    }

    private void assertRunConfigUnavailable(RunConfigUnavailableException failure) {
        assertThat(failure).isNotNull();
        assertThat(failure.getErrorCode()).isEqualTo("RUN_CONFIG_UNAVAILABLE");
    }

    @FunctionalInterface
    private interface ThrowingCall {
        void run() throws Exception;
    }
}
