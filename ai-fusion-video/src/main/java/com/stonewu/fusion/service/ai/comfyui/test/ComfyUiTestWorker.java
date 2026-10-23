package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiOutputResolver;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowDocumentService;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiNativeClient;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaJobOrigin;
import com.stonewu.fusion.service.media.model.MediaResource;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Component
@RequiredArgsConstructor
@Slf4j
public class ComfyUiTestWorker {
    private final ComfyUiTestRepository runs;
    private final ComfyUiTestService service;
    private final ComfyUiTestInputs inputs;
    private final CanvasJson json;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;
    private final ComfyUiGenerationExecutor executor;
    private final ComfyUiNativeClient client;
    private final ComfyUiWorkflowDocumentService documents;
    private final ComfyUiOutputResolver resolver;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy mediaAccess;
    private final MediaProbe probe;
    private final MediaIntakeService intake;
    private final String owner = "comfy-test-" + UUID.randomUUID();
    private final ExecutorService workers = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore slots = new Semaphore(4);
    private final Set<MediaJob.Claim> active = ConcurrentHashMap.newKeySet();

    @Scheduled(fixedDelay = 500)
    public void dispatch() {
        while (slots.tryAcquire()) {
            Optional<ComfyUiTestRun> next;
            try {
                next = runs.claim(owner);
            } catch (RuntimeException failure) {
                slots.release();
                throw failure;
            }
            if (next.isEmpty()) {
                slots.release();
                return;
            }
            var run = next.get();
            active.add(run.job().claim());
            try {
                workers.execute(() -> {
                    try {
                        execute(run);
                    } finally {
                        active.remove(run.job().claim());
                        slots.release();
                    }
                });
            } catch (RuntimeException stopped) {
                active.remove(run.job().claim());
                slots.release();
                if (!workers.isShutdown()) {
                    throw stopped;
                }
                return;
            }
        }
    }

    @Scheduled(fixedDelay = 20_000)
    public void heartbeat() {
        active.forEach(runs::heartbeat);
    }

    @PreDestroy
    public void stop() {
        workers.shutdownNow();
    }

    public boolean runOne() {
        var next = runs.claim(owner);
        if (next.isEmpty()) {
            return false;
        }
        execute(next.get());
        return true;
    }

    private void execute(ComfyUiTestRun run) {
        var claim = run.job().claim();
        try {
            if (run.job().cancelRequested() && (run.submittedAt() == null || run.result() != null)) {
                runs.finish(claim, "CANCELLED", "试运行已取消", false);
                return;
            }
            if (!run.job().deadline().isAfter(jobs.now())) {
                runs.finish(claim, run.submittedAt() == null ? "FAILED" : "NEEDS_REVIEW",
                        "自动等待已到期，请核实远端状态后继续", run.submittedAt() != null && run.result() == null);
                return;
            }
            fence.requireWorkflowTester(run.job().context().actor());
            var api = service.requireApi(run.apiConfigId(), run.apiConfigHash());
            var snapshot = json.read(json.write(run.job().input()), ComfyUiTestSnapshot.class);
            var context = snapshot.execution(api);
            if (run.submittedAt() == null) {
                jobs.check(claim);
                var prepared = executor.prepare(context, "test-" + run.job().id(),
                        inputs.resolve(snapshot, run.job().context().actor()));
                service.requireApi(run.apiConfigId(), run.apiConfigHash());
                runs.begin(claim, prepared);
                try {
                    executor.submit(prepared);
                } catch (RuntimeException failure) {
                    var rejected = GenerationProviderFailure.find(failure);
                    if (rejected == null) {
                        throw failure;
                    }
                    runs.finish(claim, "FAILED", rejected.getMessage(), false);
                    return;
                }
                runs.acknowledge(run.job().id(), prepared.promptId());
                runs.waiting(claim, "POLLING", 2, null);
                return;
            }
            if (run.result() != null) {
                ingest(run, snapshot);
                return;
            }
            if (run.job().cancelRequested() && !run.cancelSent() && runs.markCancelSent(claim)) {
                // cancelled=true 只表示请求已发出；后续仍查询实际终态。
                executor.cancel(context, run.promptId());
            }
            var result = executor.query(context, run.promptId());
            if (result.completed()) {
                runs.received(claim, result);
                var latest = runs.get(run.job().id());
                if (latest.job().cancelRequested()) {
                    runs.finish(claim, "CANCELLED", "远端已结束，试运行已取消", false);
                } else {
                    ingest(latest, snapshot);
                }
            } else if (result.failed()) {
                runs.finish(claim, result.status().equals("cancelled") ? "CANCELLED" : "FAILED",
                        result.status().equals("cancelled") ? "远端已确认取消" : "ComfyUI 工作流执行失败", false);
            } else {
                runs.waiting(claim, "POLLING", Math.min(30, 2 + run.pollAttempts() / 4), null);
            }
        } catch (Exception failure) {
            if (Thread.currentThread().isInterrupted()) {
                return;
            }
            var current = runs.get(claim.id());
            if (current.job().terminal()) {
                return;
            }
            String message = failure.getMessage() == null ? "工作流试运行失败" : failure.getMessage();
            if (message.length() > 3000) {
                message = message.substring(0, 3000);
            }
            try {
                if (current.submittedAt() == null) {
                    runs.finish(claim, "FAILED", message, false);
                } else if (current.result() != null) {
                    if (failure instanceof MediaException media && media.getErrorCode()
                            .equals("MEDIA_INGEST_IN_PROGRESS")) {
                        runs.waiting(claim, "INGESTING", 10, message);
                    } else {
                        runs.finish(claim, "NEEDS_REVIEW", "远端已完成，结果保存需要处理：" + message, false);
                    }
                } else if (current.consecutiveErrors() >= 7 || (failure instanceof MediaException media
                        && (media.getCode() == 403 || media.getErrorCode().startsWith("COMFY_TEST_PROVIDER")))) {
                    runs.finish(claim, "NEEDS_REVIEW", "远端状态需要核实：" + message, true);
                } else {
                    runs.waiting(claim, "POLLING", Math.min(60, 2 << Math.min(5, current.consecutiveErrors())),
                            message);
                }
            } catch (MediaException lost) {
                if (!lost.getErrorCode().equals("MEDIA_JOB_FENCED")) {
                    throw lost;
                }
            }
            log.warn("ComfyUI 试运行阶段未完成: id={}, phase={}", claim.id(), current.job().phase());
        }
    }

    private void ingest(ComfyUiTestRun run, ComfyUiTestSnapshot snapshot) throws IOException {
        var claim = run.job().claim();
        var api = service.requireApi(run.apiConfigId(), run.apiConfigHash());
        var bindings = documents.parseOutputBindings(snapshot.workflow().getModelType(),
                snapshot.version().getApiWorkflowJson(),
                snapshot.version().getOutputBindingsJson());
        var outputs = resolver.resolve(run.result(), bindings);
        for (int index = 0; index < outputs.size(); index++) {
            var current = jobs.check(claim);
            if (index < current.checkpoint().path("outputs").size()) {
                continue;
            }
            var output = outputs.get(index);
            String slot = "comfy-output:" + index;
            var previous = resources.request(current.context().actor().userId(), "job:" + current.id() + ":" + slot);
            MediaResource resource;
            if (previous.isPresent() && previous.get().resource().state().equals("READY")) {
                resource = previous.get().resource();
                mediaAccess.ready(resource, current.context().actor());
            } else {
                try (var file = client.downloadOutput(api, output.filename(), output.subfolder(), output.type())) {
                    var facts = probe.inspect(file.path(), file.contentType());
                    if (!output.mediaType().equals("file") && !facts.kind().name()
                            .equalsIgnoreCase(output.mediaType())) {
                        throw MediaException.invalid("试运行输出的真实文件类型与绑定不符");
                    }
                    resource = intake.job(file.path(), "工作流试运行-" + index + "." + file.extension(),
                            facts.mimeType(), slot,
                            null, current.context(), new MediaJobOrigin(claim));
                }
            }
            runs.output(claim, index,
                    new ComfyUiStoredOutput(resource.content().kind().name().toLowerCase(Locale.ROOT), output.role(),
                            MediaReference.of(resource.id()), resource.content().byteSize()));
        }
        service.requireApi(run.apiConfigId(), run.apiConfigHash());
        runs.finish(claim, "SUCCEEDED", null, false);
    }
}
