package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.ai.vo.AgentSkillImportReqVO;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentMcpServerService;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentUserMcpRuntimeRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentSkillImportService;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStateCleanupPolicyService;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceConfigService;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceMigrationService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentConfigurationControllerSkillImportTests {

    private final AgentSkillImportService importService = mock(AgentSkillImportService.class);
    private final AgentConfigurationController controller = new AgentConfigurationController(
            mock(AgentWorkspaceConfigService.class),
            mock(AgentWorkspaceMigrationService.class),
            mock(AgentUserSkillService.class),
            importService,
            mock(AgentMcpServerService.class),
            mock(AgentUserMcpRuntimeRegistry.class),
            mock(AgentStateCleanupPolicyService.class));

    @AfterEach
    void clearSecurityContext() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void previewUsesCurrentUserId() {
        authenticate(42L);
        List<MultipartFile> files = List.of(new MockMultipartFile(
                "files", "skills.zip", "application/zip", new byte[]{1}));
        List<String> paths = List.of("skills.zip");
        AgentSkillImportService.ImportPreview preview = new AgentSkillImportService.ImportPreview(
                "ZIP", 0, 0, List.of(), List.of(), List.of());
        when(importService.preview(42L, files, paths)).thenReturn(preview);

        CommonResult<AgentSkillImportService.ImportPreview> result =
                controller.previewSkillImport(files, paths);

        assertThat(result.getData()).isSameAs(preview);
        verify(importService).preview(42L, files, paths);
    }

    @Test
    void importMapsSelectionsAndUsesCurrentUserId() {
        authenticate(42L);
        List<MultipartFile> files = List.of(new MockMultipartFile(
                "files", "skills.zip", "application/zip", new byte[]{1}));
        List<String> paths = List.of("skills.zip");
        AgentSkillImportReqVO request = new AgentSkillImportReqVO(List.of(
                new AgentSkillImportReqVO.Selection(
                        "skill-a", "Skill A", AgentSkillImportService.ImportAction.CREATE)));
        List<AgentSkillImportService.ImportSelection> selections = List.of(
                new AgentSkillImportService.ImportSelection(
                        "skill-a", "Skill A", AgentSkillImportService.ImportAction.CREATE));
        AgentSkillImportService.ImportResult imported = new AgentSkillImportService.ImportResult(
                1, 0, 0, List.of("skill-a"), List.of(), List.of());
        when(importService.importSkills(42L, files, paths, selections)).thenReturn(imported);

        CommonResult<AgentSkillImportService.ImportResult> result =
                controller.importSkills(files, paths, request);

        assertThat(result.getData()).isSameAs(imported);
        verify(importService).importSkills(42L, files, paths, selections);
    }

    private void authenticate(long userId) {
        SecurityUserDetails user = new SecurityUserDetails(
                userId, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities()));
    }
}
