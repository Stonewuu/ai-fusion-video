package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.importing.*;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import static com.stonewu.fusion.service.media.importing.MediaImportData.*;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@Testcontainers(disabledWithoutDocker = false)
class MediaImportBatchIT extends MediaResourceTestContext {
    private static final AtomicLong IDS = new AtomicLong(68000);
    private static final GenerationClientScope SCOPE = new GenerationClientScope(71L, null);
    private MediaImportService imports;
    private MediaImportQueries batches;
    private MediaImportJobs jobs;
    private MediaImportWorker worker;
    private MediaImportPlacementService placement;
    private CanvasCommandService commands;
    private GenerationOutbox outbox;
    private String canvas;
    private ResourceCommandContext actor;

    @BeforeEach void prepareBatch() {
        imports = context.getBean(MediaImportService.class);
        batches = context.getBean(MediaImportQueries.class);
        jobs = context.getBean(MediaImportJobs.class);
        worker = context.getBean(MediaImportWorker.class);
        placement = context.getBean(MediaImportPlacementService.class);
        commands = context.getBean(CanvasCommandService.class);
        outbox = context.getBean(GenerationOutbox.class);
        jdbc.update("UPDATE afv_media_job SET status = 'CANCELLED' WHERE kind = 'SOURCE_IMPORT'");
        doAnswer(call -> config(((Number) call.getArgument(0)).longValue())).when(storageMapper).selectById(any());
        canvas = id();
        commands.create(new CanvasCreateCommand(canvas, "批量导入验收", null), OWNER);
        actor = ResourceCommandContext.user(OWNER, canvas);
    }

    @Test void outboxPublishesOneAcceptanceAndRepeatedSubmissionReusesTheFrozenBatch() throws Exception {
        var request = request(List.of(source(true)), new Point(50, 50));
        var first = imports.submit(canvas, request, actor);
        assertThat(first.status()).isEqualTo("QUEUED");
        assertThat(batches.active(canvas, OWNER)).extracting(Summary::id).contains(first.id());
        assertThat(worker.runOne()).isFalse();
        assertThat(imports.submit(canvas, request, actor).id()).isEqualTo(first.id());
        outbox.dispatch();
        assertThat(worker.runOne()).isTrue();
        var done = batches.get(first.id(), OWNER);
        assertThat(done.status()).isEqualTo("SUCCEEDED");
        assertThat(done.items().getFirst().resource().state()).isEqualTo("READY");
        assertThat(mapper.writeValueAsString(done)).doesNotContain("/media/", "uri", "keyset");
        outbox.dispatch();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job_event WHERE job_id = ? AND event_type = 'MEDIA_IMPORT_ACCEPTED'", Integer.class, first.id())).isEqualTo(1);
        assertThat(batches.request(canvas, request.requestId(), OWNER).id()).isEqualTo(first.id());
        assertThat(batches.active(canvas, OWNER)).extracting(Summary::id).contains(first.id());
        placement.place(first.id(), new Place(SCOPE, null, true), OWNER);
        assertThat(batches.active(canvas, OWNER)).isEmpty();
    }

    @Test void failedItemsRetryIndependentlyAndLostRetryResponseCannotStartAnotherAttempt() throws Exception {
        var missing = source(false);
        var first = imports.submit(canvas, request(List.of(source(true), missing), null), actor);
        outbox.dispatch(); worker.runOne(); worker.runOne();
        var mixed = batches.get(first.id(), OWNER);
        assertThat(mixed.status()).isEqualTo("PARTIAL_FAILED");
        String kept = mixed.items().getFirst().resource().id();
        String uri = jdbc.queryForObject("SELECT cover_url FROM afv_asset WHERE id = ?", String.class, missing.source().id());
        Files.write(directory.resolve(uri.substring(7)), png);
        var retry = new Retry(SCOPE, mixed.revision(), List.of(1));
        var resumed = imports.retry(first.id(), retry, OWNER);
        assertThat(imports.retry(first.id(), retry, OWNER).revision()).isEqualTo(resumed.revision());
        outbox.dispatch(); worker.runOne();
        var done = batches.get(first.id(), OWNER);
        assertThat(done.status()).isEqualTo("SUCCEEDED");
        assertThat(done.items().getFirst().resource().id()).isEqualTo(kept);
        assertThat(done.items().getFirst().attempt()).isEqualTo(1);
        assertThat(done.items().get(1).attempt()).isEqualTo(2);
        assertThatThrownBy(() -> imports.retry(first.id(), new Retry(SCOPE, done.revision(), List.of(0)), OWNER)).isInstanceOf(MediaException.class);
    }

    @Test void withdrawBeforeAcceptanceRejectsLateSubmissionAndKeepsOriginalScope() throws Exception {
        var request = request(List.of(source(true)), null);
        assertThat(imports.withdraw(canvas, request.requestId(), SCOPE, OWNER).accepted()).isFalse();
        assertThatThrownBy(() -> imports.submit(canvas, request, actor)).isInstanceOf(MediaException.class).hasMessageContaining("撤回");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job WHERE request_id = ?", Integer.class, request.requestId())).isZero();
        assertThatThrownBy(() -> imports.withdraw(canvas, request.requestId(), new GenerationClientScope(72L, null), OWNER)).isInstanceOf(RuntimeException.class);
    }

    @Test void cancelStopsRemainingItemsAndAutomaticPlacementButKeepsSuccessfulMedia() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true), source(true)), new Point(20, 20)), actor);
        outbox.dispatch(); worker.runOne();
        var stopped = imports.cancel(first.id(), SCOPE, OWNER);
        assertThat(stopped.status()).isEqualTo("CANCELLED");
        assertThat(stopped.items()).extracting(ItemView::status).containsExactly("SUCCEEDED", "CANCELLED");
        assertThat(worker.runOne()).isFalse();
        assertThatThrownBy(() -> placement.place(first.id(), new Place(SCOPE, null, true), OWNER)).isInstanceOf(MediaException.class);
        var placed = placement.place(first.id(), new Place(SCOPE, null, false), OWNER);
        assertThat(placed.job().items().getFirst().commandId()).isEqualTo(placed.command().commandId());
        assertThat(placed.job().items().get(1).commandId()).isNull();
    }

    @Test void reloadingAndParallelPlacementCreateOneNodeAndDeletionNeverResurrectsIt() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true)), new Point(100, 200)), actor);
        outbox.dispatch(); worker.runOne();
        try (var executor = Executors.newFixedThreadPool(2)) {
            var a = executor.submit(() -> placement.place(first.id(), new Place(SCOPE, null, true), OWNER));
            var b = executor.submit(() -> placement.place(first.id(), new Place(SCOPE, null, true), OWNER));
            assertThat(a.get(20, TimeUnit.SECONDS).job().items().getFirst().commandId()).isEqualTo(b.get(20, TimeUnit.SECONDS).job().items().getFirst().commandId());
        }
        var item = batches.get(first.id(), OWNER).items().getFirst();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_node WHERE canvas_id = ? AND deleted = FALSE", Integer.class, canvas)).isEqualTo(1);
        commands.edit(canvas, new CanvasEditCommand(id(), List.of(new CanvasGraphOperation.DeleteNode(item.nodeId(), 0, 0))), OWNER);
        placement.place(first.id(), new Place(SCOPE, null, true), OWNER);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_node WHERE canvas_id = ? AND deleted = FALSE", Integer.class, canvas)).isZero();
        assertThat(batches.get(first.id(), OWNER).items().getFirst().commandId()).isEqualTo(item.commandId());
    }

    @Test void expiredLeaseIsReclaimedAndPreviousWorkerCannotPublishOrStopTheNewWorker() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true)), null), actor);
        outbox.dispatch();
        var original = jobs.claim("before-restart").orElseThrow().claim();
        jobs.start(original);
        jdbc.update("UPDATE afv_media_job SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", first.id());
        var replacement = jobs.claim("after-restart").orElseThrow().claim();
        assertThat(replacement.epoch()).isGreaterThan(original.epoch());
        assertThatThrownBy(() -> jobs.record(original, 0, null, "TEST", "旧执行者")).isInstanceOf(MediaException.class);
        jobs.stop(original, "旧执行者不能结束接管后的任务");
        assertThat(batches.get(first.id(), OWNER).status()).isEqualTo("RUNNING");
        jobs.start(replacement);
        jobs.stop(replacement, "结束本地测试");
        assertThat(batches.get(first.id(), OWNER).status()).isEqualTo("CANCELLED");
    }

    @Test void permissionsAreCheckedAgainAfterAdmissionAndOtherAccountsCannotReadTheBatch() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true)), null), actor);
        assertThatThrownBy(() -> batches.get(first.id(), OTHER)).isInstanceOf(MediaException.class);
        outbox.dispatch();
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = 71");
        worker.runOne();
        assertThat(jdbc.queryForObject("SELECT status FROM afv_media_job WHERE id = ?", String.class, first.id())).isEqualTo("CANCELLED");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_source WHERE source_type = 'asset' AND source_id = ?", Integer.class, first.items().getFirst().source().id())).isZero();
    }

    @Test void changedSourceCannotBeImportedOrRetriedWithAnOldSnapshot() throws Exception {
        var source = source(true);
        var first = imports.submit(canvas, request(List.of(source), null), actor);
        jdbc.update("UPDATE afv_asset SET version = version + 1 WHERE id = ?", source.source().id());
        outbox.dispatch(); worker.runOne();
        var failed = batches.get(first.id(), OWNER);
        assertThat(failed.status()).isEqualTo("FAILED");
        assertThatThrownBy(() -> imports.retry(first.id(), new Retry(SCOPE, failed.revision(), List.of(0)), OWNER)).isInstanceOf(MediaException.class);
    }

    @Test void archivedCanvasAndUnavailableResultsRemainReadableSoRemainingWorkCanBeCancelled() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true), source(true)), null), actor);
        outbox.dispatch(); worker.runOne();
        var done = batches.get(first.id(), OWNER).items().getFirst().resource();
        jdbc.update("UPDATE afv_media_resource SET state = 'DELETED', version = version + 1 WHERE id = ?", done.id());
        jdbc.update("UPDATE afv_canvas SET archived = TRUE WHERE id = ?", canvas);
        var history = batches.get(first.id(), OWNER);
        assertThat(history.canControl()).isFalse();
        assertThat(history.items().getFirst().resource()).isNull();
        assertThat(history.items().getFirst().errorCode()).isEqualTo("MEDIA_IMPORT_RESULT_UNAVAILABLE");
        assertThat(imports.cancel(first.id(), SCOPE, OWNER).status()).isEqualTo("CANCELLED");
    }

    @Test void cancelDuringObjectWritingCannotPublishTheFileOrContinueWithTheNextItem() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true), source(true)), null), actor);
        outbox.dispatch();
        doAnswer(call -> {
            imports.cancel(first.id(), SCOPE, OWNER);
            return config(((Number) call.getArgument(0)).longValue());
        }).when(storageMapper).selectById(any());
        worker.runOne();
        var stopped = batches.get(first.id(), OWNER);
        assertThat(stopped.status()).isEqualTo("CANCELLED");
        assertThat(stopped.items()).allSatisfy(item -> { assertThat(item.resource()).isNull(); assertThat(item.status()).isEqualTo("CANCELLED"); });
        assertThat(worker.runOne()).isFalse();
    }

    @Test void manualPlacementRequiresAPointAndLaterCallsCannotMoveTheBatch() throws Exception {
        var first = imports.submit(canvas, request(List.of(source(true)), null), actor);
        outbox.dispatch(); worker.runOne();
        assertThatThrownBy(() -> placement.place(first.id(), new Place(SCOPE, null, false), OWNER)).isInstanceOf(MediaException.class);
        placement.place(first.id(), new Place(SCOPE, new Point(80, 160), false), OWNER);
        assertThatThrownBy(() -> placement.place(first.id(), new Place(SCOPE, new Point(400, 300), false), OWNER)).isInstanceOf(MediaException.class);
    }

    @Test void dispatchLimitsEachAccountAndYieldsBetweenItems() throws Exception {
        for (int index = 0; index < 3; index++) imports.submit(canvas, request(List.of(source(true), source(true)), null), actor);
        outbox.dispatch();
        var first = jobs.claim("one").orElseThrow();
        var second = jobs.claim("two").orElseThrow();
        assertThat(jobs.claim("three")).isEmpty();
        jobs.start(first.claim());
        jobs.record(first.claim(), 0, null, "TEST_FAILURE", "仅用于验证分项调度");
        assertThat(jobs.claim("three")).isPresent();
        assertThat(second.owner()).isEqualTo("two");
    }

    @Test void httpValidatesVersionsAndAcceptsOnlyKnownSources() throws Exception {
        var request = request(List.of(source(true)), null);
        http.perform(post("/api/canvases/" + canvas + "/media-imports").contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsString(request))).andExpect(status().isOk());
        String missingVersion = mapper.writeValueAsString(request).replace("\"version\":0,", "");
        http.perform(post("/api/canvases/" + canvas + "/media-imports").contentType(MediaType.APPLICATION_JSON)
                .content(missingVersion)).andExpect(status().isBadRequest());
        var arbitraryUrl = mapper.valueToTree(request);
        ((ObjectNode) arbitraryUrl.at("/sources/0/source")).put("url", "http://127.0.0.1/private");
        http.perform(post("/api/canvases/" + canvas + "/media-imports").contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsString(arbitraryUrl))).andExpect(status().isBadRequest());
        var url = new RequestedSource("任意网址", new MediaSourceReference(CanvasValueType.ASSET, request.sources().getFirst().source().id(), 0, "http://127.0.0.1/private"));
        assertThatThrownBy(() -> imports.submit(canvas, request(List.of(url), null), actor)).isInstanceOf(MediaException.class);
    }

    @Test void agentImportContinuesWhileWaitingAndPlacementKeepsItsInvocationAudit() throws Exception {
        String agentId = id();
        jdbc.update("""
                INSERT INTO afv_agent_run(run_id,conversation_id,user_id,canvas_scope_json,kernel_fingerprint,
                    agent_definition_snapshot_json,agent_state_session_id,status,owner_instance_id,owner_epoch,lease_until,deadline_at,started_at)
                VALUES(?,?,71,?,'test','{}',?,'RUNNING','import-agent',1,DATE_ADD(UTC_TIMESTAMP(3),INTERVAL 5 MINUTE),
                    DATE_ADD(UTC_TIMESTAMP(3),INTERVAL 1 HOUR),UTC_TIMESTAMP(3))
                """, agentId, agentId, "{\"canvasId\":\"" + canvas + "\"}", agentId);
        var intake = new ResourceCommandContext(OWNER, "agent", canvas, agentId, "source-intake", "import-agent", 1L);
        var request = request(List.of(source(true)), null);
        var accepted = imports.submit(canvas, request, intake);
        assertThat(imports.submit(canvas, request, intake).id()).isEqualTo(accepted.id());
        jdbc.update("UPDATE afv_agent_run SET status='WAITING_EXTERNAL',owner_instance_id=NULL,lease_until=NULL WHERE run_id=?", agentId);
        outbox.dispatch(); worker.runOne();
        assertThat(batches.get(accepted.id(), OWNER).status()).isEqualTo("SUCCEEDED");
        var next = new ResourceCommandContext(OWNER, "agent", canvas, agentId, "source-place", "import-agent", 2L);
        assertThatThrownBy(() -> placement.place(accepted.id(), new Place(SCOPE, new Point(40, 80), false), next)).hasMessageContaining("智能体");
        jdbc.update("UPDATE afv_agent_run SET status='RUNNING',owner_instance_id='import-agent',owner_epoch=2,lease_until=DATE_ADD(UTC_TIMESTAMP(3),INTERVAL 5 MINUTE) WHERE run_id=?", agentId);
        var result = placement.place(accepted.id(), new Place(SCOPE, new Point(40, 80), false), next);
        assertThat(result.command().changes().upsertedNodes()).hasSize(1);
        assertThat(jdbc.queryForMap("SELECT origin,agent_run_id,tool_call_id FROM afv_canvas_operation WHERE id=?", result.command().commandId()))
                .containsEntry("origin", "agent").containsEntry("agent_run_id", agentId).containsEntry("tool_call_id", "source-place");
        assertThat(placement.place(accepted.id(), new Place(SCOPE, new Point(40, 80), false), next).command()).isNull();
    }

    private Submit request(List<RequestedSource> sources, Point point) { return new Submit(id(), SCOPE, sources, point); }
    private RequestedSource source(boolean exists) throws Exception {
        long asset = IDS.incrementAndGet();
        String name = id() + ".png";
        if (exists) Files.write(directory.resolve(name), png);
        jdbc.update("INSERT INTO afv_asset(id, name, type, user_id, owner_type, owner_id, cover_url) VALUES (?, '导入源', 'image', 71, 1, 71, ?)", asset, "/media/" + name);
        return new RequestedSource("素材 " + asset, new MediaSourceReference(CanvasValueType.ASSET, asset, 0, "coverUrl"));
    }
}
