package com.stonewu.fusion.service.media;

import org.springframework.transaction.IllegalTransactionStateException;
import org.springframework.transaction.support.TransactionSynchronizationManager;

/**
 * 不创建空事务同步作用域，避免非事务查询的连接仍被 Spring 绑定到整个 I/O 方法。
 */
public final class MediaIoBoundary {
    private MediaIoBoundary() {
    }

    public static void requireOutsideTransaction() {
        if (TransactionSynchronizationManager.isActualTransactionActive()
                || TransactionSynchronizationManager.isSynchronizationActive()) {
            throw new IllegalTransactionStateException("媒体文件 I/O 必须在数据库事务及同步作用域之外执行");
        }
    }
}
