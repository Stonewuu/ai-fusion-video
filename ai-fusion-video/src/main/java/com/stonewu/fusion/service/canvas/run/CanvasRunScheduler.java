package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.common.BusinessException;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.Claim;

@Slf4j
@Component
@RequiredArgsConstructor
public class CanvasRunScheduler {
    private final CanvasRunRepository runs;
    private final CanvasRunTransactions transactions;
    private final String owner = "canvas-" + UUID.randomUUID();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(8);
    private final Map<String, Claim> active = new ConcurrentHashMap<>();

    @Scheduled(fixedDelay = 500)
    public void tick() {
        for (String id : runs.activeRuns()) {
            if (!slots.tryAcquire()) {
                return;
            }
            Claim claim;
            try {
                claim = transactions.schedule(id, owner);
            } catch (BusinessException denied) {
                slots.release();
                if (denied.getCode() == 403 || denied.getCode() == 404) {
                    transactions.inaccessible(id);
                } else {
                    log.warn("画布调度暂未完成 [runId={}]: {}", id, denied.getMessage());
                }
                continue;
            } catch (RuntimeException failure) {
                slots.release();
                log.warn("画布调度事务失败 [runId={}]", id, failure);
                continue;
            }
            if (claim == null) {
                slots.release();
                continue;
            }
            active.put(claim.id(), claim);
            workers.submit(() -> execute(claim));
        }
    }

    private void execute(Claim claim) {
        try {
            transactions.step(claim);
        } catch (Exception failure) {
            log.warn("画布节点执行未提交 [nodeRunId={}]", claim.id(), failure);
            try {
                transactions.failed(claim, message(failure));
            } catch (RuntimeException fenced) {
                log.debug("画布节点已由其他执行者处理 [nodeRunId={}]", claim.id(), fenced);
            }
        } finally {
            active.remove(claim.id(), claim);
            slots.release();
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void heartbeat() {
        active.values().forEach(runs::heartbeat);
    }

    private String message(Exception failure) {
        String text = failure.getMessage();
        return text == null ? "节点执行失败" : text.substring(0, Math.min(text.length(), 2000));
    }

    @PreDestroy
    public void close() {
        workers.shutdownNow();
    }
}
