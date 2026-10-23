package com.stonewu.fusion.service.resource;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class ResourceEventProjectionScheduler {
    private final ResourceEventProjectionService projection;

    @Scheduled(fixedDelayString = "${app.canvas.resource-event-poll-ms:500}", initialDelay = 2000)
    public void publish() {
        try {
            for (int index = 0; index < 50 && projection.publishNext(); index++) {
            }
        } catch (RuntimeException failure) {
            // 事务未完成的投递会回滚，下一轮仍从数据库发件箱领取。
            log.error("资源事件投递失败，将从持久化记录继续", failure);
        }
    }
}
