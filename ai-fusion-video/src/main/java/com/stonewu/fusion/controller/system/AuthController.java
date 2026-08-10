package com.stonewu.fusion.controller.system;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.system.vo.LoginRespVO;
import com.stonewu.fusion.controller.system.vo.UserRespVO;
import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.security.TokenService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.data.redis.core.StringRedisTemplate;

import com.stonewu.fusion.controller.system.vo.LoginReqVO;
import com.stonewu.fusion.controller.system.vo.RegisterReqVO;
import com.stonewu.fusion.controller.system.vo.ProfileUpdateReqVO;
import com.stonewu.fusion.controller.system.vo.ChangePasswordReqVO;
import com.stonewu.fusion.controller.system.vo.PasswordResetRequestVO;
import com.stonewu.fusion.controller.system.vo.PasswordResetSubmitVO;
import com.stonewu.fusion.service.team.TeamService;
import com.stonewu.fusion.service.system.UserService;
import com.stonewu.fusion.service.system.MailService;
import com.stonewu.fusion.service.system.SystemConfigService;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.crypto.SecureUtil;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.stonewu.fusion.common.CommonResult.success;

/**
 * 认证控制器
 */
@Tag(name = "认证管理")
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
@Slf4j
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final TokenService tokenService;
    private final UserService userService;
    private final TeamService teamService;
    private final StringRedisTemplate stringRedisTemplate;
    private final MailService mailService;
    private final SystemConfigService systemConfigService;

    @PostMapping("/login")
    @Operation(summary = "登录")
    public CommonResult<LoginRespVO> login(@Valid @RequestBody LoginReqVO reqVO) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(reqVO.getUsername(), reqVO.getPassword()));
        SecurityUserDetails userDetails = (SecurityUserDetails) authentication.getPrincipal();
        Long currentTeamId = teamService.getCurrentTeamIdByUser(userDetails.getUserId());
        TokenService.TokenPair tokenPair = tokenService.createToken(userDetails.getUserId(), userDetails.getUsername(),
                currentTeamId);

        User user = userService.getById(userDetails.getUserId());
        return success(buildLoginResp(tokenPair, user));
    }

    @PostMapping("/register")
    @Operation(summary = "注册")
    public CommonResult<LoginRespVO> register(@Valid @RequestBody RegisterReqVO reqVO) {
        if (!reqVO.getPassword().equals(reqVO.getConfirmPassword())) {
            throw new BusinessException(400, "两次输入的密码不一致");
        }
        User user = userService.register(reqVO.getUsername(), reqVO.getPassword(), reqVO.getNickname());
        Long currentTeamId = teamService.getCurrentTeamIdByUser(user.getId());
        TokenService.TokenPair tokenPair = tokenService.createToken(user.getId(), user.getUsername(), currentTeamId);
        return success(buildLoginResp(tokenPair, user));
    }

    @PostMapping("/refresh")
    @Operation(summary = "刷新令牌")
    public CommonResult<LoginRespVO> refresh(@Valid @RequestBody RefreshTokenReqVO reqVO) {
        TokenService.TokenPair tokenPair = tokenService.refreshAccessToken(reqVO.getRefreshToken());
        if (tokenPair == null) {
            return CommonResult.error(401, "刷新令牌无效或已过期");
        }

        // 获取用户信息
        Long userId = tokenService.getUserIdFromToken(tokenPair.getAccessToken());
        User user = userService.getById(userId);
        return success(buildLoginResp(tokenPair, user));
    }

    @PostMapping("/logout")
    @Operation(summary = "登出")
    public CommonResult<Boolean> logout(HttpServletRequest request) {
        String token = getTokenFromRequest(request);
        if (StringUtils.hasText(token)) {
            tokenService.removeToken(token);
        }
        return success(true);
    }

    @GetMapping("/user-info")
    @Operation(summary = "获取当前用户信息")
    public CommonResult<UserRespVO> getUserInfo() {
        SecurityUserDetails userDetails = (SecurityUserDetails) SecurityContextHolder.getContext()
                .getAuthentication().getPrincipal();
        User user = userService.getById(userDetails.getUserId());
        List<Role> roles = userService.getUserRoles(user.getId());

        UserRespVO respVO = new UserRespVO();
        respVO.setId(user.getId());
        respVO.setUsername(user.getUsername());
        respVO.setNickname(user.getNickname());
        respVO.setAvatar(user.getAvatar());
        respVO.setEmail(user.getEmail());
        respVO.setPhone(user.getPhone());
        respVO.setStatus(user.getStatus());
        respVO.setCreateTime(user.getCreateTime());
        respVO.setRoles(roles.stream().map(Role::getCode).toList());
        return success(respVO);
    }

    @PutMapping("/profile")
    @Operation(summary = "更新当前用户的个人资料")
    public CommonResult<Boolean> updateProfile(@Valid @RequestBody ProfileUpdateReqVO reqVO) {
        SecurityUserDetails userDetails = (SecurityUserDetails) SecurityContextHolder.getContext()
                .getAuthentication().getPrincipal();
        userService.updateProfile(userDetails.getUserId(), reqVO.getNickname(), reqVO.getEmail(), reqVO.getPhone());
        return success(true);
    }

    @PutMapping("/change-password")
    @Operation(summary = "修改当前用户密码")
    public CommonResult<Boolean> changePassword(@Valid @RequestBody ChangePasswordReqVO reqVO) {
        SecurityUserDetails userDetails = (SecurityUserDetails) SecurityContextHolder.getContext()
                .getAuthentication().getPrincipal();
        userService.changePassword(userDetails.getUserId(), reqVO.getOldPassword(), reqVO.getNewPassword());
        return success(true);
    }

    /**
     * 构建登录响应
     */
    private LoginRespVO buildLoginResp(TokenService.TokenPair tokenPair, User user) {
        return LoginRespVO.builder()
                .accessToken(tokenPair.getAccessToken())
                .refreshToken(tokenPair.getRefreshToken())
                .expiresIn(tokenPair.getExpiresIn())
                .userId(user.getId())
                .username(user.getUsername())
                .nickname(user.getNickname())
                .build();
    }

    private String getTokenFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken)) {
            if (bearerToken.startsWith("Bearer ")) {
                return bearerToken.substring(7);
            }
            return bearerToken;
        }
        return request.getParameter("access_token");
    }

    /**
     * 刷新令牌请求
     */
    @Data
    public static class RefreshTokenReqVO {
        @NotBlank(message = "刷新令牌不能为空")
        private String refreshToken;
    }

    @PostMapping("/reset-password/request")
    @Operation(summary = "申请密码找回")
    public CommonResult<String> requestResetPassword(@Valid @RequestBody PasswordResetRequestVO reqVO) {
        User user = userService.getByUsernameOrEmail(reqVO.getUsernameOrEmail());
        if (user == null) {
            throw new BusinessException(400, "用户不存在");
        }

        String type = reqVO.getType();
        if ("email".equalsIgnoreCase(type)) {
            if (StrUtil.isBlank(user.getEmail())) {
                throw new BusinessException(400, "该用户未绑定邮箱地址");
            }
            // 校验邮箱服务器配置
            String host = systemConfigService.getValue("mail_smtp_host");
            if (StrUtil.isBlank(host)) {
                throw new BusinessException(400, "系统邮箱服务未配置，请使用日志方式找回或联系管理员");
            }

            // 生成 token
            String token = IdUtil.fastSimpleUUID();
            String key = "auth:reset-token:email:" + token;
            stringRedisTemplate.opsForValue().set(key, String.valueOf(user.getId()), 24, TimeUnit.HOURS);

            // 发送邮件
            String siteBaseUrl = systemConfigService.getSiteBaseUrl();
            if (StrUtil.isBlank(siteBaseUrl)) {
                throw new BusinessException(500, "系统未配置站点公网地址，请联系管理员配置后再试");
            }

            String resetUrl = siteBaseUrl + "/forgot-password?token=" + token + "&type=email";
            String emailContent = String.format(
                    "<div style='font-family: sans-serif; padding: 20px; max-width: 600px; margin: 0 auto; border: 1px solid #eee; border-radius: 8px;'>"
                            +
                            "  <h2 style='color: #333;'>融光 · AI视频创作平台密码重置</h2>" +
                            "  <p>您好 %s，</p>" +
                            "  <p>我们收到了重置您账户密码的请求。请点击下面的链接来重置您的密码：</p>" +
                            "  <p style='margin: 30px 0;'><a href='%s' style='background-color: #000; color: #fff; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block;'>重置密码</a></p>"
                            +
                            "  <p>若无法点击上述按钮，请复制以下链接并在浏览器中打开：</p>" +
                            "  <p style='word-break: break-all; color: #666;'>%s</p>" +
                            "  <p style='color: #999; font-size: 12px; margin-top: 30px;'>该链接将在 24 小时内有效。如果您没有请求重置密码，请忽略此邮件。</p>"
                            +
                            "</div>",
                    user.getNickname(), resetUrl, resetUrl);

            mailService.sendHtmlEmail(user.getEmail(), "【融光】密码重置申请", emailContent);
            return success("密码重置链接已发送至您的邮箱，请在 24 小时内点击重置");
        } else if ("log".equalsIgnoreCase(type)) {
            // 后台验证码只支持管理员账号使用
            if (!userService.isAdmin(user.getId())) {
                throw new BusinessException(400, "日志验证码找回方式仅支持管理员账号");
            }
            // 生成足够长的验证码并写入日志
            String rawCode = IdUtil.fastSimpleUUID() + System.currentTimeMillis();
            String code = SecureUtil.sha256(rawCode);
            String key = "auth:reset-token:log:" + code;
            stringRedisTemplate.opsForValue().set(key, String.valueOf(user.getId()), 24, TimeUnit.HOURS);

            log.warn("\n======================================================================\n" +
                    "[PASSWORD RESET] 用户 {} (ID: {}) 申请重置密码。\n" +
                    "验证码为: {}\n" +
                    "请将该验证码复制并填入前端页面以重置密码。有效期为 24 小时。\n" +
                    "======================================================================",
                    user.getUsername(), user.getId(), code);

            return success("验证码已输出至系统日志，请获取并输入");
        } else {
            throw new BusinessException(400, "不支持的找回方式");
        }
    }

    @PostMapping("/reset-password/verify")
    @Operation(summary = "验证密码找回凭证")
    public CommonResult<String> verifyResetToken(@RequestParam String token, @RequestParam String type) {
        String key;
        if ("email".equalsIgnoreCase(type)) {
            key = "auth:reset-token:email:" + token;
        } else if ("log".equalsIgnoreCase(type)) {
            key = "auth:reset-token:log:" + token;
        } else {
            throw new BusinessException(400, "不支持的找回方式");
        }

        String userIdStr = stringRedisTemplate.opsForValue().get(key);
        if (StrUtil.isBlank(userIdStr)) {
            throw new BusinessException(400, "该重置链接或验证码已失效或已过期");
        }

        User user = userService.getById(Long.parseLong(userIdStr));
        if (user == null) {
            throw new BusinessException(404, "对应的用户不存在");
        }

        return success(user.getUsername());
    }

    @PostMapping("/reset-password/submit")
    @Operation(summary = "提交密码重置")
    public CommonResult<Boolean> submitResetPassword(@Valid @RequestBody PasswordResetSubmitVO reqVO) {
        if (!reqVO.getNewPassword().equals(reqVO.getConfirmPassword())) {
            throw new BusinessException(400, "两次输入的密码不一致");
        }

        String key;
        if ("email".equalsIgnoreCase(reqVO.getType())) {
            key = "auth:reset-token:email:" + reqVO.getToken();
        } else if ("log".equalsIgnoreCase(reqVO.getType())) {
            key = "auth:reset-token:log:" + reqVO.getToken();
        } else {
            throw new BusinessException(400, "不支持的找回方式");
        }

        String userIdStr = stringRedisTemplate.opsForValue().get(key);
        if (StrUtil.isBlank(userIdStr)) {
            throw new BusinessException(400, "重置链接或验证码已失效");
        }

        Long userId = Long.parseLong(userIdStr);
        userService.resetPassword(userId, reqVO.getNewPassword());

        // 成功重置密码后，使 token 失效
        stringRedisTemplate.delete(key);

        return success(true);
    }
}
