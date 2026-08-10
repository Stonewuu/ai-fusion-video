package com.stonewu.fusion.controller.system;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.system.vo.LoginRespVO;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.team.TeamService;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.stonewu.fusion.service.system.UserService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

import static com.stonewu.fusion.common.CommonResult.success;

/**
 * 系统初始化控制器
 */
@Tag(name = "系统初始化")
@RestController
@RequestMapping("/api/system/init")
@RequiredArgsConstructor
public class SystemInitController {

    private final TokenService tokenService;
    private final UserService userService;
    private final SystemConfigService systemConfigService;
    private final TeamService teamService;

    @GetMapping("/status")
    @Operation(summary = "获取系统初始化状态")
    public CommonResult<Map<String, Boolean>> getInitStatus() {
        boolean initialized = userService.isInitialized();
        boolean allowRegister = initialized && systemConfigService.isRegistrationEnabled();
        return success(Map.of("initialized", initialized, "allowRegister", allowRegister));
    }

    @PostMapping("/setup")
    @Operation(summary = "初始化管理员账号")
    public CommonResult<LoginRespVO> setup(@Valid @RequestBody SetupReqVO reqVO) {
        User user = userService.initializeAdmin(reqVO.getUsername(), reqVO.getPassword(), reqVO.getNickname());

        // 自动登录
        Long currentTeamId = teamService.getCurrentTeamIdByUser(user.getId());
        TokenService.TokenPair tokenPair = tokenService.createToken(user.getId(), user.getUsername(), currentTeamId);
        return success(LoginRespVO.builder()
                .accessToken(tokenPair.getAccessToken())
                .refreshToken(tokenPair.getRefreshToken())
                .expiresIn(tokenPair.getExpiresIn())
                .userId(user.getId())
                .username(user.getUsername())
                .nickname(user.getNickname())
                .build());
    }

    @Data
    public static class SetupReqVO {
        @NotBlank(message = "用户名不能为空")
        @Size(min = 3, max = 32, message = "用户名长度为 3-32 位")
        private String username;

        @NotBlank(message = "密码不能为空")
        @Size(min = 6, max = 32, message = "密码长度为 6-32 位")
        private String password;

        private String nickname;
    }
}
