package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.openai.errors.OpenAIIoException;
import com.openai.errors.OpenAIServiceException;
import io.agentscope.core.model.transport.HttpTransportException;

import java.io.IOException;
import java.security.cert.CertificateException;
import java.security.cert.CertPathValidatorException;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeoutException;
import java.time.Duration;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

/** 只标记模型网络调用产生的异常；工具、持久化和权限失败不能触发模型重试。 */
public final class AgentModelCallFailure extends RuntimeException {
    private final boolean retryable;
    private final Duration retryAfter;

    public AgentModelCallFailure(Throwable cause) {
        super(cause.getMessage(), cause);
        retryAfter = requestedDelay(cause);
        retryable = (retryAfter == null || retryAfter.compareTo(Duration.ofMinutes(2)) <= 0) && transientFailure(cause);
    }

    public boolean retryable() {
        return retryable;
    }

    public Duration delay(Duration scheduled) {
        return retryAfter != null && retryAfter.compareTo(scheduled) > 0 ? retryAfter : scheduled;
    }

    public String exhaustedMessage() {
        return "模型请求未能恢复，已保留执行进度，请稍后继续。";
    }

    private static boolean transientFailure(Throwable failure) {
        GenerationProviderFailure provider = GenerationProviderFailure.find(failure);
        if (provider != null && quotaCode(provider.errorCode())) {
            return false;
        }
        OpenAIServiceException service = serviceFailure(failure);
        if (service != null && service.code().filter(AgentModelCallFailure::quotaCode).isPresent()) {
            return false;
        }
        Integer status = AgentModelHttpFailure.status(failure);
        if (status != null) {
            var response = GenerationProviderFailure.find(failure);
            String reason = response == null ? failure.getMessage() : response.reason();
            String normalized = reason == null ? "" : reason.toLowerCase(Locale.ROOT);
            if (normalized.contains("insufficient_quota") || normalized.contains("insufficient quota")
                    || normalized.contains("insufficient balance") || normalized.contains("余额不足")
                    || normalized.contains("额度用尽")
                    || normalized.contains("billing_hard_limit_reached") || normalized.contains("credit balance")) {
                return false;
            }
            return status == 408 || status == 409 || status == 429
                    || status >= 500 && status <= 599 && status != 501 && status != 505;
        }
        Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        boolean transport = false;
        for (Throwable current = failure; current != null && seen.add(current); current = current.getCause()) {
            if (current instanceof InterruptedException || current instanceof CancellationException
                    || current instanceof CertificateException || current instanceof CertPathValidatorException) {
                return false;
            }
            transport |= current instanceof IOException || current instanceof TimeoutException
                    || current instanceof HttpTransportException || current instanceof OpenAIIoException;
            if (current instanceof AgentModelResponseFailure response) {
                return retryableCode(response.code());
            }
            if (current instanceof OpenAIServiceException response) {
                return response.code().map(AgentModelCallFailure::retryableCode).orElse(false);
            }
        }
        return transport;
    }

    private static boolean retryableCode(String code) {
        return "server_error".equals(code) || "rate_limit_exceeded".equals(code)
                || "model_overloaded".equals(code) || "server_is_overloaded".equals(code);
    }

    private static boolean quotaCode(String code) {
        return "insufficient_quota".equals(code) || "billing_hard_limit_reached".equals(code);
    }

    private static OpenAIServiceException serviceFailure(Throwable failure) {
        Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Throwable current = failure; current != null && seen.add(current); current = current.getCause()) {
            if (current instanceof OpenAIServiceException service) {
                return service;
            }
        }
        return null;
    }

    private static Duration requestedDelay(Throwable failure) {
        GenerationProviderFailure provider = GenerationProviderFailure.find(failure);
        OpenAIServiceException service = serviceFailure(failure);
        String header = provider != null ? provider.retryAfter()
                : service != null ? service.headers().values("retry-after").stream().findFirst().orElse(null) : null;
        if (header != null) {
            try {
                Duration delay = header.strip().matches("[0-9]+")
                        ? Duration.ofSeconds(Long.parseLong(header.strip()))
                        : Duration.between(Instant.now(), ZonedDateTime.parse(header, DateTimeFormatter.RFC_1123_DATE_TIME).toInstant());
                if (!delay.isNegative()) {
                    return delay;
                }
            } catch (NumberFormatException | DateTimeParseException invalidHeader) {
                // 外部响应头格式无效时，仍使用平台既定的重试间隔。
            }
        }
        return null;
    }
}
