package com.stonewu.fusion.database;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.BufferedReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Test;

class FlywayMigrationNamingTests {

    private static final Pattern VERSIONED_NAME = Pattern.compile(
            "^V1\\.(\\d+)\\.(\\d+)\\.(\\d+)\\.(\\d+)__[a-z0-9]+(?:_[a-z0-9]+)*\\.sql$");

    private static final Pattern BASELINE_NAME = Pattern.compile(
            "^B1\\.(\\d+)\\.(\\d+)\\.(\\d+)\\.(\\d+)__baseline_[a-z0-9]+(?:_[a-z0-9]+)*\\.sql$");

    private static final Set<String> LEGACY_MIGRATIONS = Set.of(
            "V1__init_mysql.sql",
            "V1.0.2.0.1__ai_model_unique_by_api_config.sql",
            "V1.0.2.0.2__ai_model_unique_with_deleted_id.sql",
            "V1.0.2.0.3__api_config_openai_append_v1_path.sql",
            "V1.0.2.1.1__api_config_app_secret_to_text.sql",
            "V1.0.3.0.1__api_config_proxy_settings.sql",
            "V1.0.4.0.1__expand_script_and_agent_message_text_columns.sql",
            "V1.0.5.0.1__init_default_team_and_register_config.sql",
            "V1.0.5.0.2__add_episode_compose_video.sql",
            "V1.0.6.0.1__ai_model_semantic_metadata.sql",
            "V1.0.6.1.1__expand_generation_platform_task_id.sql",
            "V1.0.6.1.2__storage_provider_profiles.sql",
            "V1.0.6.1.3__storyboard_frame_references.sql",
            "V1.0.6.1.4__storyboard_episode_script_link.sql",
            "V1.0.6.1.5__agent_run_and_event.sql");

    private static final Map<String, Integer> IMMUTABLE_MIGRATION_CHECKSUMS = Map.ofEntries(
            Map.entry("V1__init_mysql.sql", 218698920),
            Map.entry("V1.0.2.0.1__ai_model_unique_by_api_config.sql", -231554560),
            Map.entry("V1.0.2.0.2__ai_model_unique_with_deleted_id.sql", 1635232783),
            Map.entry("V1.0.2.0.3__api_config_openai_append_v1_path.sql", 1533618198),
            Map.entry("V1.0.2.1.1__api_config_app_secret_to_text.sql", 12603820),
            Map.entry("V1.0.3.0.1__api_config_proxy_settings.sql", 1958148727),
            Map.entry("V1.0.4.0.1__expand_script_and_agent_message_text_columns.sql", -621559851),
            Map.entry("V1.0.5.0.1__init_default_team_and_register_config.sql", 985925379),
            Map.entry("V1.0.5.0.2__add_episode_compose_video.sql", -191031826),
            Map.entry("V1.0.6.0.1__ai_model_semantic_metadata.sql", -726201579),
            Map.entry("V1.0.6.1.1__expand_generation_platform_task_id.sql", 931670906),
            Map.entry("V1.0.6.1.2__storage_provider_profiles.sql", -502781525),
            Map.entry("V1.0.6.1.3__storyboard_frame_references.sql", 1795138386),
            Map.entry("V1.0.6.1.4__storyboard_episode_script_link.sql", 209920478),
            Map.entry("V1.0.6.1.5__agent_run_and_event.sql", -1261062418),
            Map.entry("V1.1.0.0.0__api_config_protocol_defaults.sql", -941361809),
            Map.entry("V1.1.0.0.1__replace_model_family_with_capability_preset.sql", -1320253802));

    @Test
    void migrationsUseProductVersionAndContinuousSequence() throws Exception {
        URL migrationResource = getClass().getClassLoader().getResource("db/migration");
        assertNotNull(migrationResource, "db/migration resource must exist");

        List<String> migrationFiles;
        try (Stream<Path> paths = Files.list(Path.of(migrationResource.toURI()))) {
            migrationFiles = paths
                    .map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith(".sql"))
                    .sorted()
                    .toList();
        }

        assertTrue(migrationFiles.containsAll(LEGACY_MIGRATIONS),
                "applied legacy migrations must not be renamed or removed");

        Map<String, List<Integer>> sequencesByProductVersion = new HashMap<>();
        Set<String> versionedFlywayVersions = new HashSet<>();
        List<String> baselineFiles = new ArrayList<>();
        for (String migrationFile : migrationFiles) {
            if (LEGACY_MIGRATIONS.contains(migrationFile)) {
                continue;
            }

            if (migrationFile.startsWith("B")) {
                baselineFiles.add(migrationFile);
                continue;
            }

            Matcher matcher = VERSIONED_NAME.matcher(migrationFile);
            assertTrue(matcher.matches(), () -> "invalid Flyway migration name: " + migrationFile);

            int major = Integer.parseInt(matcher.group(1));
            int minor = Integer.parseInt(matcher.group(2));
            int patch = Integer.parseInt(matcher.group(3));
            int sequence = Integer.parseInt(matcher.group(4));
            assertTrue(major >= 1, () -> "new migrations must target product version 1.0.0 or later: "
                    + migrationFile);

            String productVersion = major + "." + minor + "." + patch;
            versionedFlywayVersions.add("1." + productVersion + "." + sequence);
            sequencesByProductVersion
                    .computeIfAbsent(productVersion, ignored -> new ArrayList<>())
                    .add(sequence);
        }

        for (String baselineFile : baselineFiles) {
            Matcher matcher = BASELINE_NAME.matcher(baselineFile);
            assertTrue(matcher.matches(), () -> "invalid Flyway baseline name: " + baselineFile);

            int major = Integer.parseInt(matcher.group(1));
            assertTrue(major >= 1, () -> "new baselines must target product version 1.0.0 or later: "
                    + baselineFile);

            String baselineFlywayVersion = "1."
                    + matcher.group(1) + "."
                    + matcher.group(2) + "."
                    + matcher.group(3) + "."
                    + matcher.group(4);
            assertTrue(versionedFlywayVersions.contains(baselineFlywayVersion),
                    () -> "baseline version must match an existing V migration: " + baselineFile);
        }

        for (Map.Entry<String, List<Integer>> entry : sequencesByProductVersion.entrySet()) {
            List<Integer> actual = entry.getValue().stream().sorted().toList();
            assertEquals(actual.size(), new HashSet<>(actual).size(),
                    () -> "duplicate migration sequence for product version " + entry.getKey());

            List<Integer> expected = new ArrayList<>();
            for (int sequence = 0; sequence < actual.size(); sequence++) {
                expected.add(sequence);
            }
            assertEquals(expected, actual,
                    () -> "migration sequence must start at 0 and remain continuous for product version "
                            + entry.getKey());
        }
    }

    @Test
    void appliedMigrationsRetainFlywayChecksums() throws Exception {
        URL migrationResource = getClass().getClassLoader().getResource("db/migration");
        assertNotNull(migrationResource, "db/migration resource must exist");
        Path migrationDirectory = Path.of(migrationResource.toURI());

        for (Map.Entry<String, Integer> entry : IMMUTABLE_MIGRATION_CHECKSUMS.entrySet()) {
            Path migration = migrationDirectory.resolve(entry.getKey());
            assertTrue(Files.isRegularFile(migration),
                    () -> "immutable migration must exist: " + entry.getKey());
            assertEquals(entry.getValue().intValue(), calculateFlywayChecksum(migration),
                    () -> "immutable migration checksum changed: " + entry.getKey());
        }
    }

    private static int calculateFlywayChecksum(Path migration) throws Exception {
        CRC32 checksum = new CRC32();
        try (BufferedReader reader = Files.newBufferedReader(migration, StandardCharsets.UTF_8)) {
            String line;
            boolean firstLine = true;
            while ((line = reader.readLine()) != null) {
                if (firstLine && !line.isEmpty() && line.charAt(0) == '\uFEFF') {
                    line = line.substring(1);
                }
                firstLine = false;
                checksum.update(line.getBytes(StandardCharsets.UTF_8));
            }
        }
        return (int) checksum.getValue();
    }
}
