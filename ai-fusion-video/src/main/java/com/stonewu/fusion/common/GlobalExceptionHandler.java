package com.stonewu.fusion.common;

import java.io.IOException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.validation.BindException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.resource.NoResourceFoundException;

/**
 * 全局异常处理器
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<CommonResult<?>> handleBusinessException(BusinessException e) {
        log.warn("业务异常: {}", e.getMessage());
        HttpStatus status = HttpStatus.resolve(e.getCode());
        if (status == null || status.is2xxSuccessful() || status.is3xxRedirection()) {
            status = HttpStatus.INTERNAL_SERVER_ERROR;
        }
        return ResponseEntity.status(status)
                .body(CommonResult.error(e.getCode(), e.getMessage()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public CommonResult<?> handleValidationException(MethodArgumentNotValidException e) {
        String message = e.getBindingResult().getAllErrors().getFirst().getDefaultMessage();
        return CommonResult.error(400, message);
    }

    @ExceptionHandler(BindException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public CommonResult<?> handleBindException(BindException e) {
        String message = e.getBindingResult().getAllErrors().getFirst().getDefaultMessage();
        return CommonResult.error(400, message);
    }

    @ExceptionHandler(BadCredentialsException.class)
    @ResponseStatus(HttpStatus.UNAUTHORIZED)
    public CommonResult<?> handleBadCredentialsException(BadCredentialsException e) {
        return CommonResult.error(401, "用户名或密码错误");
    }

    @ExceptionHandler(AccessDeniedException.class)
    @ResponseStatus(HttpStatus.FORBIDDEN)
    public CommonResult<?> handleAccessDeniedException(AccessDeniedException e) {
        return CommonResult.error(403, "没有权限访问");
    }

    @ExceptionHandler(NoResourceFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public CommonResult<?> handleNoResourceFoundException(NoResourceFoundException e) {
        log.warn("静态资源未找到: {}", e.getMessage());
        return CommonResult.error(404, "资源未找到");
    }

    /**
     * SSE / 长连接客户端断开时触发的 IOException，属于正常行为，降级为 DEBUG 日志。
     * <p>
     * 注意：SSE 端点的 Content-Type 已经是 text/event-stream，
     * 此时不能返回 CommonResult（没有匹配的 HttpMessageConverter），直接 swallow 即可。
     */
    @ExceptionHandler(IOException.class)
    public void handleIOException(IOException e) {
        String msg = e.getMessage();
        if (msg != null && (msg.contains("已建立的连接") || msg.contains("Broken pipe")
                || msg.contains("Connection reset"))) {
            log.debug("客户端连接断开（SSE/长连接正常行为）: {}", msg);
        } else {
            log.error("IO 异常", e);
        }
        // 连接已断开，无法写回响应，直接返回
    }

    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public CommonResult<?> handleException(Exception e) {
        log.error("系统异常", e);
        return CommonResult.error(500, "系统内部错误");
    }
}
