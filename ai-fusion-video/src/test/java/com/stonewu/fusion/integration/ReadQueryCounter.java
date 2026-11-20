package com.stonewu.fusion.integration;

import org.apache.ibatis.executor.Executor;
import org.apache.ibatis.mapping.MappedStatement;
import org.apache.ibatis.plugin.Interceptor;
import org.apache.ibatis.plugin.Intercepts;
import org.apache.ibatis.plugin.Invocation;
import org.apache.ibatis.plugin.Signature;
import org.apache.ibatis.session.ResultHandler;
import org.apache.ibatis.session.RowBounds;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/** 记录一次集合读取的查询数量，防止逐条详情读取重新进入集合接口。 */
@Intercepts(@Signature(type = Executor.class, method = "query",
        args = {MappedStatement.class, Object.class, RowBounds.class, ResultHandler.class}))
final class ReadQueryCounter implements Interceptor {
    private final ThreadLocal<List<String>> active = new ThreadLocal<>();

    record Measurement<T>(T value, List<String> statements) {
    }

    <T> Measurement<T> measure(Supplier<T> query) {
        List<String> statements = new ArrayList<>();
        active.set(statements);
        try {
            return new Measurement<>(query.get(), List.copyOf(statements));
        } finally {
            active.remove();
        }
    }

    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        List<String> statements = active.get();
        if (statements != null) {
            statements.add(((MappedStatement) invocation.getArgs()[0]).getId());
        }
        return invocation.proceed();
    }
}
