package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.common.BusinessException;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.mock.web.MockMultipartFile;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentSkillImportServiceTests {

    @Test
    void previewsSingleSkillWithResourcesAndPreservesScriptAsNonExecutable() throws Exception {
        AgentUserSkillService userSkills = mock(AgentUserSkillService.class);
        AgentSkillImportService service = new AgentSkillImportService(userSkills);
        Map<String, byte[]> entries = new LinkedHashMap<>();
        entries.put("pdf-processing/SKILL.md", skillMarkdown(
                "pdf-processing", "Process PDF files when a user asks about PDFs", true));
        entries.put("pdf-processing/references/guide.md", "reference".getBytes(StandardCharsets.UTF_8));
        entries.put("pdf-processing/scripts/run.py", "print('ok')".getBytes(StandardCharsets.UTF_8));
        entries.put("pdf-processing/assets/logo.png", new byte[]{(byte) 0x89, 0x50, 0x4e, 0x47});

        AgentSkillImportService.ImportPreview preview = service.preview(
                42L,
                List.of(zipUpload(entries)),
                List.of("skills.zip"));

        assertThat(preview.sourceType()).isEqualTo("ZIP");
        assertThat(preview.errors()).isEmpty();
        assertThat(preview.skills()).singleElement().satisfies(skill -> {
            assertThat(skill.name()).isEqualTo("pdf-processing");
            assertThat(skill.fileCount()).isEqualTo(4);
            assertThat(skill.hasScripts()).isTrue();
            assertThat(skill.hasReferences()).isTrue();
            assertThat(skill.hasAssets()).isTrue();
            assertThat(skill.valid()).isTrue();
            assertThat(skill.warnings()).anyMatch(value -> value.contains("不会执行脚本"));
            assertThat(skill.warnings()).anyMatch(value -> value.contains("不会自动授予工具权限"));
        });
    }

    @Test
    void previewsMultipleSkillsInsideOneWrapperDirectory() throws Exception {
        AgentUserSkillService userSkills = mock(AgentUserSkillService.class);
        AgentSkillImportService service = new AgentSkillImportService(userSkills);
        Map<String, byte[]> entries = new LinkedHashMap<>();
        entries.put("bundle/skill-a/SKILL.md", skillMarkdown("skill-a", "Use for task A", false));
        entries.put("bundle/skill-b/SKILL.md", skillMarkdown("skill-b", "Use for task B", false));

        AgentSkillImportService.ImportPreview preview = service.preview(
                42L,
                List.of(zipUpload(entries)),
                List.of("bundle.zip"));

        assertThat(preview.errors()).isEmpty();
        assertThat(preview.skills()).extracting(AgentSkillImportService.ImportCandidate::name)
                .containsExactly("skill-a", "skill-b");
    }

    @Test
    void rejectsZipSlipBeforeWritingAnything() throws Exception {
        AgentUserSkillService userSkills = mock(AgentUserSkillService.class);
        AgentSkillImportService service = new AgentSkillImportService(userSkills);
        Map<String, byte[]> entries = Map.of(
                "../evil/SKILL.md", skillMarkdown("evil", "Use for evil", false));

        assertThatThrownBy(() -> service.preview(
                42L,
                List.of(zipUpload(entries)),
                List.of("skills.zip")))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("路径穿越");
    }

    @Test
    void importsSelectedSkillsWithCreateAndReplacePolicies() throws Exception {
        AgentUserSkillService userSkills = mock(AgentUserSkillService.class);
        AgentSkillImportService service = new AgentSkillImportService(userSkills);
        when(userSkills.exists(42L, "skill-a")).thenReturn(false);
        when(userSkills.exists(42L, "skill-b")).thenReturn(true);
        when(userSkills.skillCount(42L)).thenReturn(1);
        when(userSkills.requireDisplayName(anyString())).thenAnswer(invocation -> invocation.getArgument(0));
        Map<String, byte[]> entries = new LinkedHashMap<>();
        entries.put("skill-a/SKILL.md", skillMarkdown("skill-a", "Use for task A", false));
        entries.put("skill-a/references/a.md", "A".getBytes(StandardCharsets.UTF_8));
        entries.put("skill-b/SKILL.md", skillMarkdown("skill-b", "Use for task B", false));

        AgentSkillImportService.ImportResult result = service.importSkills(
                42L,
                List.of(zipUpload(entries)),
                List.of("skills.zip"),
                List.of(
                        new AgentSkillImportService.ImportSelection(
                                "skill-a", "Skill A", AgentSkillImportService.ImportAction.CREATE),
                        new AgentSkillImportService.ImportSelection(
                                "skill-b", "Skill B", AgentSkillImportService.ImportAction.REPLACE)));

        assertThat(result.createdNames()).containsExactly("skill-a");
        assertThat(result.replacedNames()).containsExactly("skill-b");
        ArgumentCaptor<Map<String, byte[]>> files = ArgumentCaptor.forClass(Map.class);
        verify(userSkills).replaceImportedDirectory(eq(42L), eq("skill-a"), eq("Skill A"), files.capture());
        assertThat(files.getValue()).containsKeys("SKILL.md", "references/a.md");
        verify(userSkills).replaceImportedDirectory(eq(42L), eq("skill-b"), eq("Skill B"), anyMap());
    }

    @Test
    void reportsInvalidStandardNameWithoutImportingIt() throws Exception {
        AgentUserSkillService userSkills = mock(AgentUserSkillService.class);
        AgentSkillImportService service = new AgentSkillImportService(userSkills);
        Map<String, byte[]> entries = Map.of(
                "bad_name/SKILL.md", skillMarkdown("bad_name", "Use for invalid names", false));

        AgentSkillImportService.ImportPreview preview = service.preview(
                42L,
                List.of(zipUpload(entries)),
                List.of("skills.zip"));

        assertThat(preview.skills()).singleElement().satisfies(skill -> {
            assertThat(skill.valid()).isFalse();
            assertThat(skill.errors()).anyMatch(value -> value.contains("frontmatter.name"));
        });
    }

    private MockMultipartFile zipUpload(Map<String, byte[]> entries) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(output, StandardCharsets.UTF_8)) {
            for (Map.Entry<String, byte[]> entry : entries.entrySet()) {
                zip.putNextEntry(new ZipEntry(entry.getKey()));
                zip.write(entry.getValue());
                zip.closeEntry();
            }
        }
        return new MockMultipartFile("files", "skills.zip", "application/zip", output.toByteArray());
    }

    private byte[] skillMarkdown(String name, String description, boolean allowedTools) {
        String tools = allowedTools ? "allowed-tools: Read\n" : "";
        return ("---\n"
                + "name: " + name + "\n"
                + "description: " + description + "\n"
                + tools
                + "---\n"
                + "# Instructions\n\nFollow the workflow.\n")
                .getBytes(StandardCharsets.UTF_8);
    }
}
