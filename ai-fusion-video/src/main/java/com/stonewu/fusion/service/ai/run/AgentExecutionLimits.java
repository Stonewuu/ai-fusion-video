package com.stonewu.fusion.service.ai.run;

import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;

/**
 * 0 表示平台不设执行上限；只在数据库和 SDK 边界转换成其支持的无上限表示。
 */
public final class AgentExecutionLimits {
    // 既有 deadline_at 为 NOT NULL。最大数据库日期专门表示无期限，不为它建立计时器。
    public static final Instant UNLIMITED_DEADLINE = Instant.parse("9999-12-31T23:59:59.999Z");
    // AgentScope 2.0 会把 null 合并成默认五分钟，沿用 agenteam 的 SDK 边界处理。
    public static final Duration FRAMEWORK_TOOL_TIMEOUT = Duration.ofNanos(Long.MAX_VALUE);

    private AgentExecutionLimits() {
    }

    public static int frameworkSteps(int steps) {
        return steps == 0 ? Integer.MAX_VALUE : steps;
    }

    public static Instant deadlineFrom(Instant now, Duration timeout) {
        return timeout.isZero() ? UNLIMITED_DEADLINE : now.plus(timeout).truncatedTo(ChronoUnit.MILLIS);
    }

    public static LocalDateTime deadlineFrom(LocalDateTime now, Duration timeout) {
        return LocalDateTime.ofInstant(deadlineFrom(now.toInstant(ZoneOffset.UTC), timeout), ZoneOffset.UTC);
    }

    public static boolean unlimited(Instant deadline) {
        return UNLIMITED_DEADLINE.equals(deadline);
    }

    public static <T> Mono<T> untilDeadline(Mono<T> action, Instant deadline) {
        if (unlimited(deadline)) {
            return action;
        }
        return Mono.defer(() -> {
            Duration remaining = Duration.between(Instant.now(), deadline);
            if (remaining.isNegative() || remaining.isZero()) {
                return Mono.error(new IllegalStateException("智能体运行期限已过"));
            }
            return action.timeout(remaining);
        });
    }
}
