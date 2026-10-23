package com.stonewu.fusion.service.media;

import lombok.RequiredArgsConstructor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * 进程退出后也能结束过期的接收状态；文件由保留策略处理，不删除仍有引用的媒体。
 */
@Component
@RequiredArgsConstructor
public class MediaIngestRecovery {
    private final MediaRegistryTransactions registry;

    @Scheduled(fixedDelayString = "${media.ingest.recovery-interval-ms:30000}")
    public void recover() {
        registry.expireUploads();
    }
}
