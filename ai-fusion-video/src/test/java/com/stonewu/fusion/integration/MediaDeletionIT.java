package com.stonewu.fusion.integration;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaCopyService;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.mapper.media.MediaPurgeMapper;
import com.stonewu.fusion.service.media.deletion.MediaPurgeTransactions;
import com.stonewu.fusion.service.media.preview.MediaPreviewService;
import com.stonewu.fusion.service.media.preview.MediaPreviewData;
import com.stonewu.fusion.service.media.preview.MediaPreviewQueries;
import com.stonewu.fusion.service.media.preview.MediaPreviewWorker;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.deletion.MediaDeletionData;
import com.stonewu.fusion.service.media.deletion.MediaDeletionData.Mode;
import com.stonewu.fusion.service.media.deletion.MediaDeletionData.Request;
import com.stonewu.fusion.service.media.deletion.MediaDeletionService;
import com.stonewu.fusion.service.media.deletion.MediaPurgeWorker;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.springframework.http.MediaType;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.transaction.IllegalTransactionStateException;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Files;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mockingDetails;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.reset;
import static org.mockito.ArgumentMatchers.any;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 验证删除后的可见结果、引用安全、当前权限和事务外文件清理，不依赖前端隐藏按钮。 */
@Testcontainers(disabledWithoutDocker = false)
class MediaDeletionIT extends MediaResourceTestContext {
    @BeforeAll
    static void registerCopyService() {
        context.registerBean(MediaCopyService.class);
    }

    @Test
    void deletingTheOriginalAlsoUnlinksAndDeletesItsDerivedPreview() throws Exception {
        var media = upload(id(), png, "带缩略图.png", null, USER);
        var previews = context.getBean(MediaPreviewService.class);
        previews.ensure(media.id(), new MediaPreviewData.Request(null), OWNER);
        context.getBean(GenerationOutbox.class).dispatch();
        var views = context.getBean(MediaPreviewQueries.class);
        for (int index = 0; index < 10 && !views.get(media.id(), OWNER).status().equals("SUCCEEDED"); index++) {
            assertThat(context.getBean(MediaPreviewWorker.class).runOne()).isTrue();
        }
        var view = views.get(media.id(), OWNER);
        assertThat(view.status()).isEqualTo("SUCCEEDED");
        var preview = queries.get(view.previewId(), OWNER);
        assertThatThrownBy(() -> service().preview(preview.id(), OWNER))
                .isInstanceOf(MediaException.class).hasMessageContaining("原素材");
        long asset = asset(preview);
        service().remove(media.id(), request(media, Mode.PERMANENT), OWNER);
        var reads = context.getBean(ResourceReadRepository.class);
        assertThat(reads.data(reads.require(CanvasValueType.ASSET, asset, false)).path("coverUrl").isNull()).isTrue();
        assertThatThrownBy(() -> queries.get(preview.id(), OWNER)).isInstanceOf(MediaException.class);
        context.getBean(MediaPurgeWorker.class).runPending();
        assertThat(service().result(media.id(), Mode.PERMANENT).fileStatus()).isEqualTo("PURGED");
    }

    @Test
    void activePreviewBlocksPermanentDeletionButAllowsLibraryRemoval() throws Exception {
        var media = upload(id(), png, "处理中.png", null, USER);
        var request = request(media, Mode.PERMANENT);
        context.getBean(MediaPreviewService.class).ensure(media.id(), new MediaPreviewData.Request(null), OWNER);
        assertThat(service().preview(media.id(), OWNER).permanentBlockedReason()).contains("任务");
        assertThatThrownBy(() -> service().remove(media.id(), request, OWNER))
                .isInstanceOf(MediaException.class).hasMessageContaining("任务");
        assertThat(service().remove(media.id(), request(media, Mode.LIBRARY), OWNER).fileStatus()).isEqualTo("RETAINED");
        assertThat(queries.get(media.id(), OWNER).state()).isEqualTo("READY");
    }

    @Test
    void storageFailureRemainsPendingAndRetriesOutsideDatabaseTransactions() throws Exception {
        var media = upload(id(), png, "重试删除.png", null, USER);
        var row = repository.get(media.id(), false);
        var file = directory.resolve(".afv-managed").resolve(repository.blob(row.blobId()).objectKey());
        service().remove(media.id(), request(media, Mode.PERMANENT), OWNER);
        var storage = spy(blobs);
        doAnswer(invocation -> {
            assertThat(TransactionSynchronizationManager.isActualTransactionActive()).isFalse();
            assertThat(TransactionSynchronizationManager.isSynchronizationActive()).isFalse();
            throw new IOException("模拟存储暂不可用");
        }).when(storage).purge(any(), any());
        var worker = new MediaPurgeWorker(context.getBean(MediaPurgeTransactions.class), repository, storage);
        worker.runPending();
        var queue = context.getBean(MediaPurgeMapper.class);
        var pending = queue.selectById(row.blobId());
        assertThat(pending.getState()).isEqualTo("PENDING");
        assertThat(pending.getAttempts()).isEqualTo(1);
        assertThat(file).exists();
        assertThat(service().result(media.id(), Mode.PERMANENT).fileStatus()).isEqualTo("PENDING");
        assertThatThrownBy(() -> context.getBean(TransactionTemplate.class).execute(ignored -> {
            worker.runPending();
            return null;
        })).isInstanceOf(IllegalTransactionStateException.class);
        reset(storage);
        pending.setNextAttemptAt(LocalDateTime.now(ZoneOffset.UTC).minusSeconds(1));
        queue.updateById(pending);
        worker.runPending();
        assertThat(file).doesNotExist();
        assertThat(queue.selectById(row.blobId()).getState()).isEqualTo("PURGED");
    }

    @Test
    void projectRevocationPreventsDeletionReplay() throws Exception {
        var media = upload(id(), png, "项目权限.png", 71L, USER);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(71,72,2)");
        var preview = service().preview(media.id(), OTHER);
        var request = new Request(id(), preview.version(), Mode.LIBRARY, preview.referenceToken());
        service().remove(media.id(), request, OTHER);
        jdbc.update("DELETE FROM afv_project_member WHERE project_id=71 AND user_id=72");
        assertThatThrownBy(() -> service().remove(media.id(), request, OTHER)).isInstanceOf(BusinessException.class);
    }
    @Test
    void libraryRemovalPreservesExistingReferencesVersionsAndAccess() throws Exception {
        var media = upload(id(), png, "保留引用.png", null, USER);
        var canvas = canvas(media);
        int version = media.version();
        String ticket = token(tickets.browser(media.id(), OWNER, false));
        var request = request(media, Mode.LIBRARY);

        assertThat(service().remove(media.id(), request, OWNER).fileStatus()).isEqualTo("RETAINED");
        assertThat(service().remove(media.id(), request, OWNER).fileStatus()).isEqualTo("RETAINED");
        assertThat(queries.list(null, 1, 100, OWNER).items()).extracting(MediaResource::id).doesNotContain(media.id());
        assertThat(queries.get(media.id(), OWNER).version()).isEqualTo(version);
        assertThat(tickets.authorize(media.id(), ticket).resource().id()).isEqualTo(media.id());
        assertThat(graph(canvas).nodes().getFirst().config().path("resourceId").asText()).isEqualTo(media.id());
    }

    @Test
    void permanentDeletionClearsLiveReferencesRevokesTicketsAndPurgesTheFile() throws Exception {
        var media = upload(id(), png, "永久删除.png", null, USER);
        var canvas = canvas(media);
        long asset = asset(media);
        var source = graph(canvas).nodes().getFirst();
        var generator = new CanvasNode(id(), CanvasNodeKind.IMAGE_GENERATION, 1, "生成图片",
                new CanvasNodeLayout(400, 0, null, null, null, false, false),
                mapper.createObjectNode().put("prompt", "海边"), null, 0, 0);
        context.getBean(CanvasCommandService.class).edit(canvas, new CanvasEditCommand(id(), List.of(
                new CanvasGraphOperation.CreateNode(generator),
                new CanvasGraphOperation.CreateEdge(new CanvasEdge(id(), CanvasEdgeKind.EXECUTION,
                        source.id(), "media", generator.id(), "referenceImages", 0, null, 0)))), OWNER);
        String ticket = token(tickets.browser(media.id(), OWNER, false));
        var row = repository.get(media.id(), false);
        var path = directory.resolve(".afv-managed").resolve(repository.blob(row.blobId()).objectKey());
        var request = request(media, Mode.PERMANENT);

        http.perform(delete("/api/media-resources/" + media.id()).contentType(MediaType.APPLICATION_JSON)
                        .content(mapper.writeValueAsBytes(request))).andExpect(status().isOk());
        assertThat(graph(canvas).nodes().stream().filter(node -> node.id().equals(source.id())).findFirst().orElseThrow()
                .config().has("resourceId")).isFalse();
        assertThat(graph(canvas).edges()).isEmpty();
        var reads = context.getBean(ResourceReadRepository.class);
        assertThat(reads.data(reads.require(CanvasValueType.ASSET, asset, false)).path("coverUrl").isNull()).isTrue();
        assertThatThrownBy(() -> tickets.authorize(media.id(), ticket)).isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> queries.get(media.id(), OWNER)).isInstanceOf(MediaException.class).hasMessageContaining("已删除");
        assertThat(repository.retentionReference(media.id())).isPresent();
        context.getBean(MediaPurgeWorker.class).runPending();
        assertThat(Files.exists(path)).isFalse();
        assertThat(service().remove(media.id(), request, OWNER).fileStatus()).isEqualTo("PURGED");
        assertThatThrownBy(() -> context.getBean(CanvasCommandService.class).edit(canvas,
                new CanvasEditCommand(id(), List.of(new CanvasGraphOperation.PatchNode(
                        source.id(), source.configVersion() + 1, null,
                        mapper.createObjectNode().put("resourceId", media.id())))), OWNER))
                .isInstanceOf(MediaException.class);
    }

    @Test
    void changedReferencesRequireASecondConfirmationAndCannotBeDeletedByAnotherUser() throws Exception {
        var media = upload(id(), png, "重新确认.png", null, USER);
        var request = request(media, Mode.PERMANENT);
        long asset = asset(media);
        assertThatThrownBy(() -> service().remove(media.id(), request, OWNER))
                .isInstanceOf(MediaException.class).hasMessageContaining("引用已变化");
        assertThat(queries.get(media.id(), OWNER).state()).isEqualTo("READY");
        when(actors.current()).thenReturn(OTHER);
        http.perform(delete("/api/media-resources/" + media.id()).contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsBytes(request))).andExpect(status().isForbidden());
        assertThat(context.getBean(ResourceReadRepository.class)
                .data(context.getBean(ResourceReadRepository.class).require(CanvasValueType.ASSET, asset, false))
                .path("coverUrl").asText()).isEqualTo("media://" + media.id());
    }

    @Test
    void deletionDoesNotRemoveTheFileUsedByAnIndependentCopy() throws Exception {
        var media = upload(id(), png, "共享文件.png", null, USER);
        var copy = context.getBean(TransactionTemplate.class).execute(ignored ->
                context.getBean(MediaCopyService.class).copy(media.id(), media.version(), null, id(), OWNER));
        var row = repository.get(media.id(), false);
        var path = directory.resolve(".afv-managed").resolve(repository.blob(row.blobId()).objectKey());
        assertThat(service().preview(media.id(), OWNER).sharedCopies()).isEqualTo(1);
        var originalDeletion = request(media, Mode.PERMANENT);
        assertThat(service().remove(media.id(), originalDeletion, OWNER).fileStatus()).isEqualTo("SHARED");
        context.getBean(MediaPurgeWorker.class).runPending();
        assertThat(Files.exists(path)).isTrue();
        assertThat(queries.get(copy.id(), OWNER).state()).isEqualTo("READY");
        service().remove(copy.id(), request(copy, Mode.PERMANENT), OWNER);
        assertThat(service().remove(media.id(), originalDeletion, OWNER).fileStatus()).isEqualTo("PENDING");
        context.getBean(MediaPurgeWorker.class).runPending();
        assertThat(Files.exists(path)).isFalse();
    }

    @Test
    void newBusinessReferenceCannotCommitAfterPermanentDeletion() throws Exception {
        var media = upload(id(), png, "并发删除.png", null, USER);
        long asset = emptyAsset();
        var request = request(media, Mode.PERMANENT);
        var locked = new CountDownLatch(1);
        var proceed = new CountDownLatch(1);
        var once = new AtomicBoolean();
        var delegate = mockingDetails(repository).getMockCreationSettings().getDefaultAnswer();
        doAnswer(invocation -> {
            var result = delegate.answer(invocation);
            if (Thread.currentThread().getName().equals("media-delete-test") && once.compareAndSet(false, true)) {
                locked.countDown();
                assertThat(proceed.await(10, TimeUnit.SECONDS)).isTrue();
            }
            return result;
        }).when(repository).get(media.id(), true);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var deletion = executor.submit(() -> {
                Thread.currentThread().setName("media-delete-test");
                return service().remove(media.id(), request, OWNER);
            });
            assertThat(locked.await(10, TimeUnit.SECONDS)).isTrue();
            var binding = executor.submit(() -> {
                assertThatThrownBy(() -> commands().patch(new ResourcePatchCommand(id(), CanvasValueType.ASSET, asset, 0,
                        mapper.createObjectNode().put("coverUrl", "media://" + media.id())), USER))
                        .isInstanceOf(BusinessException.class);
            });
            proceed.countDown();
            assertThat(deletion.get(10, TimeUnit.SECONDS).mode()).isEqualTo(Mode.PERMANENT);
            binding.get(10, TimeUnit.SECONDS);
        } finally {
            proceed.countDown();
        }
    }

    private MediaDeletionService service() {
        return context.getBean(MediaDeletionService.class);
    }

    private Request request(MediaResource media, Mode mode) {
        var preview = service().preview(media.id(), OWNER);
        return new Request(id(), preview.version(), mode, preview.referenceToken());
    }

    private String canvas(MediaResource media) {
        String canvas = id();
        var commands = context.getBean(CanvasCommandService.class);
        commands.create(new CanvasCreateCommand(canvas, "删除验证画布", null), OWNER);
        commands.edit(canvas, new CanvasEditCommand(id(), List.of(new CanvasGraphOperation.CreateNode(
                new CanvasNode(id(), CanvasNodeKind.IMAGE, 1, "参考图",
                        new CanvasNodeLayout(0, 0, null, null, null, false, false),
                        mapper.createObjectNode().put("resourceId", media.id()), null, 0, 0)))), OWNER);
        return canvas;
    }

    private CanvasGraph graph(String id) {
        return context.getBean(CanvasGraphRepository.class).load(context.getBean(CanvasDocumentRepository.class).require(id, false));
    }

    private ResourceCommandService commands() {
        return context.getBean(ResourceCommandService.class);
    }

    private long emptyAsset() {
        var fields = mapper.createObjectNode().put("name", "删除验证角色").put("type", "character");
        fields.putObject("properties").put("gender", "unknown");
        return commands().execute(new ResourceCreateCommand(id(), CanvasValueType.ASSET, null, fields), USER)
                .result().path("id").asLong();
    }

    private long asset(MediaResource media) {
        long asset = emptyAsset();
        commands().patch(new ResourcePatchCommand(id(), CanvasValueType.ASSET, asset, 0,
                mapper.createObjectNode().put("coverUrl", "media://" + media.id())), USER);
        return asset;
    }
}
