package com.stonewu.fusion.integration;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.mapper.generation.ImageTaskMapper;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import com.stonewu.fusion.mapper.generation.VideoTaskMapper;
import com.stonewu.fusion.mapper.generation.VideoItemMapper;
import com.stonewu.fusion.infrastructure.queue.RedisTaskQueue;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationDeletionGuard;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.VideoFrameExtractor;
import com.stonewu.fusion.service.media.GenerationMediaLinks;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.media.GenerationFrameService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.media.ProviderMediaFetcher;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.media.model.MediaSourceType;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.reset;

@Testcontainers(disabledWithoutDocker = false)
class GenerationMediaIT extends MediaResourceTestContext {
    static ImageGenerationService images;
    static VideoGenerationService videos;
    static GenerationMediaService media;
    static GenerationExecutionContexts executions;

    @BeforeAll
    static void registerGeneration() throws Exception {
        context.getBeanFactory().registerSingleton("redisTaskQueue", mock(RedisTaskQueue.class));
        context.register(GenerationMediaLinks.class, GenerationMediaService.class, ProviderMediaFetcher.class,
                GenerationFrameService.class, VideoFrameExtractor.class, GenerationDeletionGuard.class,
                ImageGenerationService.class, VideoGenerationService.class);
        images = context.getBean(ImageGenerationService.class);
        videos = context.getBean(VideoGenerationService.class);
        media = context.getBean(GenerationMediaService.class);
        executions = context.getBean(GenerationExecutionContexts.class);
    }

    @BeforeEach
    void storage() {
        doAnswer(call -> config(((Number) call.getArgument(0)).longValue())).when(storageMapper).selectById(any());
    }

    @Test
    void actualGenerationMappersVersionAndLinkRegisteredResultsWithoutPublicCopies() throws Exception {
        var task = imageTask();
        var item = images.createItem(ImageItem.builder().taskId(task.getId()).build());
        images.updateStatus(task.getId(), 1, null);
        var input = Files.createTempFile(directory, "generated-", ".png");
        Files.write(input, png);
        String reference = media.file(GenerationKind.IMAGE, task.getId(), "item-" + item.getId() + "-image", input,
                MediaKind.IMAGE);
        assertThat(reference).startsWith("media://");
        assertThat(media.file(GenerationKind.IMAGE, task.getId(), "item-" + item.getId() + "-image", input,
                MediaKind.IMAGE)).isEqualTo(reference);
        item.setImageUrl(reference);
        item.setStatus(1);
        images.updateItem(item);
        assertThat(item.getVersion()).isEqualTo(1);
        assertThat(images.listItems(task.getId()).getFirst().getVersion()).isEqualTo(1);
        String id = MediaReference.id(reference);
        var source = new MediaSourceReference(MediaSourceType.IMAGE_ITEM, item.getId(), 1, "imageUrl");
        assertThat(queries.sources(id, OWNER)).contains(source);
        task.setSuccessCount(1);
        images.update(task);
        assertThat(images.getById(task.getId()).getStatus()).isEqualTo(1);
        images.updateStatus(task.getId(), 2, null);
        assertThat(context.getBean(MediaSourceImportService.class).importSource(source, USER).id()).isEqualTo(id);
        assertThatThrownBy(
                () -> media.file(GenerationKind.IMAGE, task.getId(), "late", input, MediaKind.IMAGE)).isInstanceOf(
                MediaException.class);
        assertThat(images.getByTaskId(task.getTaskId()).getStatus()).isEqualTo(2);
        assertThat(mapper.writeValueAsString(images.getById(task.getId()))).doesNotContain("executionContextJson",
                "ownerInstanceId");
    }

    @Test
    void cancellationDuringStorageCannotPublishMediaOrOverwriteTerminalTaskState() throws Exception {
        var task = imageTask();
        images.updateStatus(task.getId(), 1, null);
        var input = Files.createTempFile(directory, "cancel-", ".png");
        Files.write(input, png);
        var cancelled = new AtomicBoolean();
        doAnswer(call -> {
            if (cancelled.compareAndSet(false, true)) {
                images.updateStatus(task.getId(), 3, "用户取消");
            }
            return config(((Number) call.getArgument(0)).longValue());
        }).when(storageMapper).selectById(any());
        assertThatThrownBy(() -> media.file(GenerationKind.IMAGE, task.getId(), "cancelled", input, MediaKind.IMAGE))
                .isInstanceOf(MediaException.class);
        assertThat(jdbc.queryForObject("SELECT state FROM afv_media_resource WHERE request_id = ?",
                String.class, "generation:image:" + task.getId() + ":cancelled")).isEqualTo("FAILED");
        assertThatThrownBy(() -> images.updateStatus(task.getId(), 2, null)).isInstanceOf(BusinessException.class);
        assertThat(images.getById(task.getId()).getErrorMsg()).isEqualTo("用户取消");
    }

    @Test
    void sourceAuditFailureRollsBackItemAndCanReplayPreviouslyReceivedMedia() throws Exception {
        var task = imageTask();
        var item = images.createItem(ImageItem.builder().taskId(task.getId()).build());
        images.updateStatus(task.getId(), 1, null);
        var input = Files.createTempFile(directory, "rollback-", ".png");
        Files.write(input, png);
        String reference = media.file(GenerationKind.IMAGE, task.getId(), "stable", input, MediaKind.IMAGE);
        item.setImageUrl(reference);
        item.setStatus(1);
        doThrow(new IllegalStateException("source audit unavailable")).when(repository).bindSource(any(), any());
        assertThatThrownBy(() -> images.updateItem(item)).isInstanceOf(IllegalStateException.class);
        assertThat(images.listItems(task.getId()).getFirst().getVersion()).isZero();
        reset(repository);
        var retry = images.listItems(task.getId()).getFirst();
        retry.setImageUrl(media.file(GenerationKind.IMAGE, task.getId(), "stable", input, MediaKind.IMAGE));
        retry.setStatus(1);
        images.updateItem(retry);
        assertThat(queries.sources(MediaReference.id(reference), OWNER)).hasSize(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_blob WHERE resource_id = ?", Integer.class,
                MediaReference.id(reference))).isEqualTo(1);
    }

    @Test
    void staleItemsAndLateProviderWritesAreRejectedByRealDatabase() {
        var task = imageTask();
        var item = images.createItem(ImageItem.builder().taskId(task.getId()).build());
        images.updateStatus(task.getId(), 1, null);
        var stale = images.listItems(task.getId()).getFirst();
        item.setPlatformTaskId("accepted");
        images.updateItem(item);
        stale.setPlatformTaskId("stale");
        assertThatThrownBy(() -> images.updateItem(stale)).isInstanceOf(BusinessException.class);
        images.updateStatus(task.getId(), 3, "用户取消");
        var late = images.listItems(task.getId()).getFirst();
        late.setImageUrl("https://example.com/late.png");
        late.setStatus(1);
        assertThatThrownBy(() -> images.updateItem(late)).isInstanceOf(BusinessException.class);
        assertThat(images.listItems(task.getId()).getFirst().getImageUrl()).isNull();
    }

    @Test
    void unspecifiedVideoAudioFlagSurvivesDatabaseRoundTrip() {
        var omitted = VideoTask.builder().taskId(id()).prompt("无声音开关的模型").projectId(71L).build();
        executions.prepare(omitted, USER);
        videos.create(omitted);
        assertThat(videos.getById(omitted.getId()).getGenerateAudio()).isNull();
        var explicit = VideoTask.builder().taskId(id()).prompt("明确关闭声音").projectId(71L).generateAudio(false)
                .build();
        executions.prepare(explicit, USER);
        videos.create(explicit);
        assertThat(videos.getById(explicit.getId()).getGenerateAudio()).isFalse();
    }

    @Test
    void realVideoAndBothFramesArePrivateRegisteredAndVersioned() throws Exception {
        var task = VideoTask.builder().taskId(id()).prompt("视频测试").projectId(71L).build();
        executions.prepare(task, USER);
        videos.create(task);
        var item = videos.createItem(VideoItem.builder().taskId(task.getId()).build());
        videos.updateStatus(task.getId(), 1, null);
        Path file = Files.createTempFile(directory, "source-video-", ".mp4");
        var process = new ProcessBuilder("ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f",
                "lavfi",
                "-i", "testsrc2=size=64x48:rate=4", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                file.toString())
                .redirectError(ProcessBuilder.Redirect.DISCARD).redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
        assertThat(process.waitFor(30, TimeUnit.SECONDS)).isTrue();
        assertThat(process.exitValue()).isZero();
        item.setVideoUrl(media.file(GenerationKind.VIDEO, task.getId(), "video", file, MediaKind.VIDEO));
        item.setStatus(1);
        videos.updateItem(item);
        var frames = context.getBean(GenerationFrameService.class).extract(task, item, true, true);
        assertThat(frames.firstFrameUrl()).startsWith("media://");
        assertThat(frames.lastFrameUrl()).startsWith("media://").isNotEqualTo(frames.firstFrameUrl());
        item.setFirstFrameUrl(frames.firstFrameUrl());
        item.setLastFrameUrl(frames.lastFrameUrl());
        videos.updateItem(item);
        videos.updateStatus(task.getId(), 2, null);
        var source = new MediaSourceReference(MediaSourceType.VIDEO_ITEM, item.getId(), item.getVersion(),
                "firstFrameUrl");
        assertThat(context.getBean(MediaSourceImportService.class).importSource(source, USER).id()).isEqualTo(
                MediaReference.id(frames.firstFrameUrl()));
        assertThatThrownBy(() -> executions.requireRead(GenerationKind.VIDEO, task.getId(), OTHER)).isInstanceOf(
                BusinessException.class);
    }

    @Test
    void historicalGenerationImportsUseExactOwnerFieldAndVersionWithoutInventingExecutionContext() throws Exception {
        var task = imageTask();
        var item = images.createItem(ImageItem.builder().taskId(task.getId()).build());
        images.updateStatus(task.getId(), 1, null);
        String name = id() + ".png";
        Files.write(directory.resolve(name), png);
        item.setImageUrl("/media/" + name);
        item.setStatus(1);
        images.updateItem(item);
        images.updateStatus(task.getId(), 2, null);
        jdbc.update("UPDATE afv_image_task SET execution_context_json = NULL WHERE id = ?", task.getId());
        var source = new MediaSourceReference(MediaSourceType.IMAGE_ITEM, item.getId(), item.getVersion(), "imageUrl");
        var imported = context.getBean(MediaSourceImportService.class).importSource(source, USER);
        assertThat(imported.projectId()).isEqualTo(71);
        assertThat(imported.ownerId()).isEqualTo(71);
        assertThat(imported.content().kind()).isEqualTo(MediaKind.IMAGE);
        assertThatThrownBy(() -> context.getBean(MediaSourceImportService.class).importSource(source,
                ResourceCommandContext.user(OTHER, null))).isInstanceOf(BusinessException.class);
        jdbc.update("UPDATE afv_image_item SET version = version + 1 WHERE id = ?", item.getId());
        assertThatThrownBy(() -> context.getBean(MediaSourceImportService.class).importSource(source, USER))
                .isInstanceOf(BusinessException.class);
    }

    @Test
    void cancelledQueuedItemsCanEndButNeverAcceptLateOutput() {
        var task = imageTask();
        var item = images.createItem(ImageItem.builder().taskId(task.getId()).build());
        images.updateStatus(task.getId(), 3, "用户取消");
        item.setStatus(2);
        item.setErrorMsg("用户取消");
        images.updateItem(item);
        assertThat(images.listItems(task.getId()).getFirst().getStatus()).isEqualTo(2);
        item.setImageUrl("https://example.com/late.png");
        item.setStatus(1);
        assertThatThrownBy(() -> images.updateItem(item)).isInstanceOf(BusinessException.class);
        assertThat(images.listItems(task.getId()).getFirst().getImageUrl()).isNull();
    }

    @Test
    void personalHistoryHidesTasksWhoseProjectGrantWasRevoked() {
        long project = 78291L;
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (?, '撤权生成项目', 1, 72)",
                project);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, 71, 3)", project);
        var task = ImageTask.builder().taskId(id()).prompt("协作生成").projectId(project).build();
        executions.prepare(task, USER);
        images.create(task);
        assertThat(images.pageByUser(OWNER, executions.readableProjects(OWNER), 1, 100).getList())
                .extracting(ImageTask::getId).contains(task.getId());
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = ?", project);
        assertThat(images.pageByUser(OWNER, executions.readableProjects(OWNER), 1, 100).getList())
                .extracting(ImageTask::getId).doesNotContain(task.getId());
        assertThatThrownBy(() -> executions.requireRead(GenerationKind.IMAGE, task.getId(), OWNER)).isInstanceOf(
                BusinessException.class);
    }

    private ImageTask imageTask() {
        var task = ImageTask.builder().taskId(UUID.randomUUID().toString()).prompt("真实生成").projectId(71L).build();
        executions.prepare(task, USER);
        return images.create(task);
    }
}
