package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.ai.vo.AgentSkillImportReqVO;
import com.stonewu.fusion.controller.ai.vo.AgentSkillSaveReqVO;
import com.stonewu.fusion.controller.ai.vo.AgentMcpServerRespVO;
import com.stonewu.fusion.controller.ai.vo.AgentMcpServerSaveReqVO;
import com.stonewu.fusion.controller.ai.vo.AgentMcpTestRespVO;
import com.stonewu.fusion.controller.ai.vo.AgentWorkspaceConfigRespVO;
import com.stonewu.fusion.controller.ai.vo.AgentWorkspaceMigrateReqVO;
import com.stonewu.fusion.controller.ai.vo.AgentStateCleanupPolicyRespVO;
import com.stonewu.fusion.controller.ai.vo.AgentStateCleanupPolicySaveReqVO;
import com.stonewu.fusion.entity.ai.AgentWorkspaceConfig;
import com.stonewu.fusion.entity.ai.AgentWorkspaceMigration;
import com.stonewu.fusion.entity.ai.AgentMcpServer;
import com.stonewu.fusion.entity.ai.AgentStateCleanupPolicy;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentMcpServerService;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentUserMcpRuntimeRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentSkillImportService;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceConfigService;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceMigrationService;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStateCleanupPolicyService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.time.LocalDateTime;
import java.time.Instant;
import java.time.ZoneOffset;

import static com.stonewu.fusion.common.CommonResult.success;
import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

@Tag(name = "智能体配置")
@RestController
@RequestMapping("/api/ai/agent-config")
@RequiredArgsConstructor
public class AgentConfigurationController {

    private final AgentWorkspaceConfigService workspaceConfigService;
    private final AgentWorkspaceMigrationService migrationService;
    private final AgentUserSkillService userSkillService;
    private final AgentSkillImportService skillImportService;
    private final AgentMcpServerService mcpServerService;
    private final AgentUserMcpRuntimeRegistry userMcpRuntimeRegistry;
    private final AgentStateCleanupPolicyService stateCleanupPolicyService;

    @GetMapping("/state-cleanup")
    @Operation(summary = "获取 AgentState 清理配置")
    public CommonResult<AgentStateCleanupPolicyRespVO> stateCleanupPolicy() {
        AgentStateCleanupPolicy policy = stateCleanupPolicyService.getCurrent();
        return success(toStateCleanupResponse(policy));
    }

    @PutMapping("/state-cleanup")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "更新 AgentState 清理配置")
    public CommonResult<AgentStateCleanupPolicyRespVO> updateStateCleanupPolicy(
            @Valid @RequestBody AgentStateCleanupPolicySaveReqVO request) {
        AgentStateCleanupPolicy policy = stateCleanupPolicyService.update(
                request.cleanupIntervalDays(), request.retentionDays());
        return success(toStateCleanupResponse(policy));
    }

    @GetMapping("/workspace")
    @Operation(summary = "获取智能体工作空间配置")
    public CommonResult<AgentWorkspaceConfigRespVO> workspace() {
        AgentWorkspaceConfig config = workspaceConfigService.getCurrent();
        AgentWorkspaceConfigService.WorkspaceUsage usage = workspaceConfigService.usage();
        return success(new AgentWorkspaceConfigRespVO(
                config.getBackendType(),
                config.getStorageConfigId(),
                config.getLocalPath(),
                config.getMigrationStatus(),
                config.getActiveMigrationId(),
                usage.entryCount(),
                usage.contentBytes(),
                migrationService.latest()));
    }

    @PostMapping("/workspace/test")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "测试智能体工作空间目标存储")
    public CommonResult<Boolean> testWorkspace(@Valid @RequestBody AgentWorkspaceMigrateReqVO request) {
        migrationService.test(request.backendType(), request.storageConfigId(), request.localPath());
        return success(true);
    }

    @PostMapping("/workspace/migrations")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "迁移并切换智能体工作空间存储")
    public CommonResult<Long> migrateWorkspace(@Valid @RequestBody AgentWorkspaceMigrateReqVO request) {
        return success(migrationService.start(
                request.backendType(), request.storageConfigId(), request.localPath()));
    }

    @GetMapping("/workspace/migrations/{id}")
    @Operation(summary = "获取智能体工作空间迁移进度")
    public CommonResult<AgentWorkspaceMigration> migration(@PathVariable Long id) {
        return success(migrationService.get(id));
    }

    @PostMapping("/workspace/migrations/{id}/rollback")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "回滚已完成的智能体工作空间迁移")
    public CommonResult<Boolean> rollbackWorkspace(@PathVariable Long id) {
        migrationService.rollback(id);
        return success(true);
    }

    @PostMapping("/workspace/migrations/{id}/dismiss-failure")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "解除失败迁移对工作空间写入的锁定")
    public CommonResult<Boolean> dismissWorkspaceMigrationFailure(@PathVariable Long id) {
        migrationService.dismissFailure(id);
        return success(true);
    }

    @GetMapping("/skills")
    @Operation(summary = "获取当前用户的自定义 Skill")
    public CommonResult<List<AgentUserSkillService.UserSkill>> skills() {
        return success(userSkillService.list(requireCurrentUserId()));
    }

    @PutMapping("/skills")
    @Operation(summary = "创建或更新当前用户的自定义 Skill")
    public CommonResult<AgentUserSkillService.UserSkill> saveSkill(
            @Valid @RequestBody AgentSkillSaveReqVO request) {
        return success(userSkillService.save(
                requireCurrentUserId(),
                request.originalName(),
                request.name(),
                request.displayName(),
                request.description(),
                request.content()));
    }

    @PostMapping(value = "/skills/import/preview", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "预检当前用户要导入的 Skill 包")
    public CommonResult<AgentSkillImportService.ImportPreview> previewSkillImport(
            @RequestPart("files") List<MultipartFile> files,
            @RequestParam("paths") List<String> paths) {
        return success(skillImportService.preview(requireCurrentUserId(), files, paths));
    }

    @PostMapping(value = "/skills/import", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "导入当前用户选择的 Skill")
    public CommonResult<AgentSkillImportService.ImportResult> importSkills(
            @RequestPart("files") List<MultipartFile> files,
            @RequestParam("paths") List<String> paths,
            @Valid @RequestPart("request") AgentSkillImportReqVO request) {
        List<AgentSkillImportService.ImportSelection> selections = request.selections().stream()
                .map(selection -> new AgentSkillImportService.ImportSelection(
                        selection.rootPath(),
                        selection.displayName(),
                        selection.action()))
                .toList();
        return success(skillImportService.importSkills(
                requireCurrentUserId(), files, paths, selections));
    }

    @DeleteMapping("/skills/{name}")
    @Operation(summary = "删除当前用户的自定义 Skill")
    public CommonResult<Boolean> deleteSkill(@PathVariable String name) {
        userSkillService.delete(requireCurrentUserId(), name);
        return success(true);
    }

    @GetMapping("/mcp")
    @Operation(summary = "获取当前用户的自定义 MCP 服务")
    public CommonResult<List<AgentMcpServerRespVO>> mcpServers() {
        long userId = requireCurrentUserId();
        return success(mcpServerService.list(userId).stream()
                .map(mcpServerService::toResponse)
                .toList());
    }

    @PutMapping("/mcp")
    @Operation(summary = "创建或更新当前用户的自定义 MCP 服务")
    public CommonResult<AgentMcpServerRespVO> saveMcpServer(
            @Valid @RequestBody AgentMcpServerSaveReqVO request) {
        long userId = requireCurrentUserId();
        AgentMcpServer saved = mcpServerService.save(userId, request);
        userMcpRuntimeRegistry.invalidate(userId);
        return success(mcpServerService.toResponse(saved));
    }

    @DeleteMapping("/mcp/{id}")
    @Operation(summary = "删除当前用户的自定义 MCP 服务")
    public CommonResult<Boolean> deleteMcpServer(@PathVariable Long id) {
        long userId = requireCurrentUserId();
        mcpServerService.delete(userId, id);
        userMcpRuntimeRegistry.invalidate(userId);
        return success(true);
    }

    @PostMapping("/mcp/{id}/test")
    @Operation(summary = "测试当前用户的 MCP 服务并发现工具")
    public CommonResult<AgentMcpTestRespVO> testMcpServer(@PathVariable Long id) {
        long userId = requireCurrentUserId();
        AgentMcpServer server = mcpServerService.requireOwned(id, userId);
        AgentMcpTestRespVO result = userMcpRuntimeRegistry.test(server);
        mcpServerService.recordTest(userId, id, result.success(), result.message());
        return success(result);
    }

    private AgentStateCleanupPolicyRespVO toStateCleanupResponse(
            AgentStateCleanupPolicy policy) {
        return new AgentStateCleanupPolicyRespVO(
                policy.getCleanupIntervalDays(),
                policy.getRetentionDays(),
                toInstant(policy.getNextCleanupAt()),
                toInstant(policy.getLastCleanupAt()));
    }

    private Instant toInstant(LocalDateTime value) {
        return value == null ? null : value.toInstant(ZoneOffset.UTC);
    }
}
