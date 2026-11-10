package com.stonewu.fusion.service.generation.audio;

import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.service.ai.google.GoogleAudioProtocol;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** 供应商未返回音频时结束任务，不把空响应显示为可重试保存的文件。 */
class AudioTaskWorkerTests {
    @Test
    void emptyProviderOutputDoesNotTriggerAnotherGenerationOrAStorageRetry() {
        var repository = mock(AudioTaskRepository.class);
        var runner = mock(AudioTaskRunner.class);
        var task = new AudioTask();
        task.setId("missing-audio");
        task.setOutputJson("received response");
        when(repository.claim(anyString())).thenReturn(Optional.of(task), Optional.empty());
        when(runner.save(task)).thenThrow(new GoogleAudioProtocol.InvalidOutput("Google 未返回音频"));
        var worker = new AudioTaskWorker(repository, runner);
        try {
            worker.dispatch();
            verify(repository, timeout(2000)).finish(task, "NO_OUTPUT", null, "Google 未返回音频");
            verify(runner, never()).submit(any());
        } finally {
            worker.close();
        }
    }
}
