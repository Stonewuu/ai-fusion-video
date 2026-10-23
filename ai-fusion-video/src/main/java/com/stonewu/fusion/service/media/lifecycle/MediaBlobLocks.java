package com.stonewu.fusion.service.media.lifecycle;

import com.stonewu.fusion.mapper.canvas.CanvasExportBlobMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DataAccessException;
import org.springframework.integration.redis.util.RedisLockRegistry;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.function.BooleanSupplier;

/**
 * 文件 I/O 使用可续租的跨实例对象锁；数据库只参与短暂的状态与业务校验。
 */
@Component
@RequiredArgsConstructor
public class MediaBlobLocks {
    public enum Domain {
        MEDIA, DELIVERY
    }

    private final MediaBlobMapper blobs;
    private final CanvasExportBlobMapper exports;
    private final RedisLockRegistry registry;

    @FunctionalInterface
    public interface Work<T> {
        T run(String state) throws IOException;
    }

    public <T> T exclusive(String id, Domain domain, Work<T> work) throws IOException {
        String key = domain.name() + ":" + id;
        var lock = registry.obtain(key);
        try {
            if (!lock.tryLock()) {
                throw new IOException("媒体文件正在处理，请稍后重试");
            }
        } catch (DataAccessException failure) {
            throw new IOException("媒体文件锁暂时不可用", failure);
        }
        Throwable original = null;
        try {
            String state = domain == Domain.MEDIA ? blobs.selectById(id).getState() : exports.selectById(id).getState();
            T result = work.run(state);
            // 续租失败或锁已失效时，不允许调用方继续登记成功。
            registry.renewLock(key);
            return result;
        } catch (IOException | RuntimeException | Error failure) {
            original = failure;
            throw failure;
        } finally {
            try {
                lock.unlock();
            } catch (RuntimeException failure) {
                if (original != null) {
                    original.addSuppressed(failure);
                } else {
                    throw new IOException("媒体文件锁已失效，本次操作未完成", failure);
                }
            }
        }
    }

    public boolean purge(String id, Domain domain, BooleanSupplier allowed, Work<Boolean> work) throws IOException {
        return exclusive(id, domain, state -> {
            // 复核回调的短事务在返回时提交，物理删除不持有任务行或文件行锁。
            if (!allowed.getAsBoolean()) {
                return false;
            }
            return work.run(state);
        });
    }
}
