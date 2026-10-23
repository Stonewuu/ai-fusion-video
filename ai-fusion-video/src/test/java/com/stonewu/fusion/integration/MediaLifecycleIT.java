package com.stonewu.fusion.integration;

import com.stonewu.fusion.service.media.MediaCopyService;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaCryptor;
import com.stonewu.fusion.mapper.canvas.CanvasExportBlobMapper;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportCleanup;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.lifecycle.AgentWorkspaceMediaIndexer;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLocks;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLocks.Domain;
import com.stonewu.fusion.mapper.media.MediaBlobCleanupMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.service.media.lifecycle.MediaCleanupWorker;
import com.stonewu.fusion.service.media.lifecycle.MediaOrphanCleanup;
import com.stonewu.fusion.service.media.model.MediaBlob;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.transaction.IllegalTransactionStateException;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.integration.redis.util.RedisLockRegistry;
import org.testcontainers.junit.jupiter.Testcontainers;
import software.amazon.awssdk.services.s3.model.S3Exception;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;

import java.io.IOException;
import java.io.OutputStream;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletionException;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.argThat;

@Testcontainers
class MediaLifecycleIT extends MediaResourceTestContext {
    private MediaOrphanCleanup jobs;
    private MediaCleanupWorker worker;

    record Orphan(MediaResource media, MediaBlob blob, Path plaintext) {
    }

    @BeforeEach
    void cleanupServices() {
        if (context.getBeanNamesForType(MediaOrphanCleanup.class).length == 0) {
            context.registerBean("lifecycleCleanup", MediaOrphanCleanup.class,
                    () -> new MediaOrphanCleanup(context.getBean(MediaBlobCleanupMapper.class),
                            context.getBean(MediaBlobMapper.class), context.getBean(MediaResourceMapper.class),
                            context.getBean(AgentWorkspaceEntryMapper.class), repository, 30));
            context.registerBean("lifecycleWorker", MediaCleanupWorker.class,
                    () -> new MediaCleanupWorker(mock(AgentWorkspaceMediaIndexer.class),
                            context.getBean(MediaOrphanCleanup.class), blobs, mock(CanvasExportCleanup.class)));
            context.registerBean(MediaCopyService.class);
        }
        jobs = context.getBean(MediaOrphanCleanup.class);
        worker = context.getBean(MediaCleanupWorker.class);
        // 先登记上个用例留下但未调度的孤儿，再统一延后，避免串扰本用例领取顺序。
        jobs.discover();
        jdbc.update(
                "UPDATE afv_media_blob_cleanup SET next_attempt_at=TIMESTAMPADD(DAY,1,UTC_TIMESTAMP(6)) WHERE status<>'PURGED'");
        jdbc.update("DELETE FROM afv_agent_workspace_entry WHERE namespace_key='lifecycle-test'");
    }

    @Test
    void expiredFailedIntakeIsDeletedOnceAndStaleWriterCannotRecreateCiphertext() throws Exception {
        var orphan = orphan(true);
        Path file = directory.resolve(".afv-managed").resolve(orphan.blob().objectKey());
        assertThat(file).exists();
        jobs.discover();
        assertThat(worker.runOne()).isTrue();
        assertThat(file).doesNotExist();
        assertThat(state(orphan)).isEqualTo("PURGED");
        assertThat(jdbc.queryForObject("SELECT LENGTH(keyset_bytes) FROM afv_media_blob WHERE id=?", Integer.class,
                orphan.blob().id())).isZero();
        assertThat(jdbc.queryForObject("SELECT attempts FROM afv_media_blob_cleanup WHERE blob_id=?", Integer.class,
                orphan.blob().id())).isEqualTo(1);
        assertThat(worker.runOne()).isFalse();
        assertThatThrownBy(() -> blobs.write(orphan.blob(), orphan.plaintext())).isInstanceOf(IOException.class)
                .hasMessageContaining("不能重新写入旧对象");
        assertThat(file).doesNotExist();
        assertThat(repository.get(orphan.media().id(), false).resource().state()).isEqualTo("FAILED");
    }

    @Test
    void recentFailedFilesRemainUntilTheRetentionDeadline() throws Exception {
        var orphan = orphan(false);
        jobs.discover();
        assertThat(worker.runOne()).isFalse();
        assertThat(state(orphan)).isEqualTo("ORPHAN");
        assertThat(directory.resolve(".afv-managed").resolve(orphan.blob().objectKey())).exists();
        assertThat(
                jdbc.queryForObject("SELECT eligible_at>UTC_TIMESTAMP(6) FROM afv_media_blob_cleanup WHERE blob_id=?",
                        Boolean.class, orphan.blob().id())).isTrue();
    }

    @Test
    void anyReadyAliasProtectsTheSharedFileEvenWhenTheOriginalIsUnavailable() throws Exception {
        var source = upload(id(), png, "副本来源.png", null, USER);
        var alias = context.getBean(TransactionTemplate.class).execute(status -> context.getBean(MediaCopyService.class)
                .copy(source.id(), source.version(), 71L, id(), OWNER));
        var blob = repository.blob(repository.get(source.id(), false).blobId());
        jdbc.update("UPDATE afv_media_resource SET state='FAILED' WHERE id=?", source.id());
        jdbc.update(
                "UPDATE afv_media_blob SET state='ORPHAN',orphaned_at=TIMESTAMPADD(DAY,-31,UTC_TIMESTAMP(6)) WHERE id=?",
                blob.id());
        jobs.discover();
        assertThat(worker.runOne()).isFalse();
        assertThat(directory.resolve(".afv-managed").resolve(blob.objectKey())).exists();
        assertThat(repository.get(alias.id(), false).resource().state()).isEqualTo("READY");
        assertThat(jdbc.queryForObject("SELECT status FROM afv_media_blob_cleanup WHERE blob_id=?", String.class,
                blob.id())).isEqualTo("BLOCKED");
    }

    @Test
    void AgentTextReferencesRetainFilesWithoutGrantingAnotherUserAccess() throws Exception {
        var orphan = orphan(true);
        jdbc.update(
                "INSERT INTO afv_agent_message(conversation_id,role,content,message_order) VALUES(?,'assistant',?,1)",
                id(), "预览 [图片](media://" + orphan.media().id() + ")");
        assertThat(jdbc.queryForList("SELECT anchor_kind FROM afv_media_reference WHERE media_id=?", String.class,
                orphan.media().id())).contains("AGENT_MESSAGE");
        jobs.discover();
        assertThat(worker.runOne()).isFalse();
        assertThat(state(orphan)).isEqualTo("ORPHAN");
        assertThatThrownBy(() -> queries.get(orphan.media().id(), OTHER)).isInstanceOfSatisfying(MediaException.class,
                failure -> assertThat(failure.getCode()).isEqualTo(403));
    }

    @Test
    void businessBindingRetainsAnOtherwiseExpiredFailedOrphan() throws Exception {
        var orphan = orphan(true);
        jdbc.update(
                "INSERT INTO afv_media_binding(resource_type,resource_id,field_name,media_id,resource_version,bound_by) VALUES('ASSET',71,?, ?,1,71)",
                id(), orphan.media().id());
        jobs.discover();
        assertThat(worker.runOne()).isFalse();
        assertThat(state(orphan)).isEqualTo("ORPHAN");
        assertThat(directory.resolve(".afv-managed").resolve(orphan.blob().objectKey())).exists();
    }

    @Test
    void targetedRetentionLookupMatchesTheAuthoritativeViewForEveryReferenceShape() throws Exception {
        String[] shapes = {
                "{\"mediaId\":\"%s\"}",
                "{\"items\":[{\"resourceId\":\"%s\"}]}",
                "{\"image\":\"media://%s\"}",
                "[\"media://%s\"]",
                "{\"message\":\"预览 [图片](media://%s) 并保留历史\"}"
        };
        for (String shape : shapes) {
            var orphan = orphan(true);
            jdbc.update(
                    "INSERT INTO afv_agent_message(conversation_id,role,content,references_json,message_order) VALUES(?,'assistant','引用验收',?,1)",
                    id(), shape.formatted(orphan.media().id()));
            assertThat(repository.retentionReference(orphan.media().id())).contains("AGENT_MESSAGE");
            assertThat(jdbc.queryForList("SELECT anchor_kind FROM afv_media_reference WHERE media_id=? LIMIT 1",
                    String.class, orphan.media().id()))
                    .containsExactly("AGENT_MESSAGE");
        }
        var plain = orphan(true);
        jdbc.update(
                "INSERT INTO afv_agent_message(conversation_id,role,content,references_json,message_order) VALUES(?,'assistant',?,JSON_OBJECT('untyped',?),1)",
                id(), plain.media().id(), plain.media().id());
        assertThat(repository.retentionReference(plain.media().id())).isEmpty();
        assertThat(
                jdbc.queryForList("SELECT anchor_kind FROM afv_media_reference WHERE media_id=? LIMIT 1", String.class,
                        plain.media().id())).isEmpty();
    }

    @Test
    void unindexedExternalWorkspaceBlocksCleanupAndItsIndexRetainsTheReferencedFile() throws Exception {
        var orphan = orphan(true);
        String hash = "a".repeat(64);
        jdbc.update(
                "INSERT INTO afv_agent_workspace_entry(namespace_hash,namespace_key,item_hash,item_key,backend_type,content_ref,content_sha256,content_size) VALUES(?,'lifecycle-test',?,'file','local','objects/reference.json',?,30)",
                hash, hash, hash);
        jobs.discover();
        assertThat(worker.runOne()).isFalse();
        jdbc.update(
                "UPDATE afv_agent_workspace_entry SET media_references_json=JSON_ARRAY(?) WHERE namespace_key='lifecycle-test'",
                "media://" + orphan.media().id());
        assertThat(jdbc.queryForList("SELECT anchor_kind FROM afv_media_reference WHERE media_id=?", String.class,
                orphan.media().id())).contains("AGENT_WORKSPACE");
        assertThat(worker.runOne()).isFalse();
        assertThat(state(orphan)).isEqualTo("ORPHAN");
    }

    @Test
    void slowEncryptionLeavesDatabaseFreeAndAnotherInstanceDefersCleanup() throws Exception {
        Path source = directory.resolve("slow-" + id() + ".png");
        Files.write(source, png);
        var facts = context.getBean(MediaProbe.class).inspect(source, "image/png");
        var claim = registry.admit(id(), null, "慢写入.png", facts, USER);
        var blob = repository.blob(claim.blobId());
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var cryptor = context.getBean(MediaCryptor.class);
        doAnswer(call -> {
            pauseIo(entered, release);
            return call.callRealMethod();
        }).when(cryptor).encrypt(any(Path.class), any(OutputStream.class), any(byte[].class), eq(blob.resourceId()));
        // 独立注册表模拟另一应用实例，不能只依靠当前 JVM 中的 ReentrantLock。
        var remoteRegistry = new RedisLockRegistry(context.getBean(RedisConnectionFactory.class), "afv:media:blob");
        var remoteLocks = new MediaBlobLocks(context.getBean(MediaBlobMapper.class),
                context.getBean(CanvasExportBlobMapper.class), remoteRegistry);
        var store = new MediaBlobStore(storageMapper, context.getBean(S3StorageConfigResolver.class),
                context.getBean(S3ClientFactory.class), cryptor, remoteLocks);
        var remoteWorker = new MediaCleanupWorker(mock(AgentWorkspaceMediaIndexer.class), jobs, store,
                mock(CanvasExportCleanup.class));
        var writer = CompletableFuture.supplyAsync(() -> {
            try {
                return blobs.write(blob, source);
            } catch (IOException failure) {
                throw new CompletionException(failure);
            }
        });
        try {
            assertThat(entered.await(5, TimeUnit.SECONDS)).isTrue();
            CompletableFuture.runAsync(() -> registry.fail(claim)).get(5, TimeUnit.SECONDS);
            jdbc.update("UPDATE afv_media_blob SET orphaned_at=TIMESTAMPADD(DAY,-31,UTC_TIMESTAMP(6)) WHERE id=?",
                    blob.id());
            jobs.discover();
            assertThat(remoteWorker.runOne()).isTrue();
            assertThat(jdbc.queryForObject("SELECT state FROM afv_media_blob WHERE id=?", String.class, blob.id()))
                    .isEqualTo("ORPHAN");
            assertThat(jdbc.queryForObject("SELECT status FROM afv_media_blob_cleanup WHERE blob_id=?", String.class,
                    blob.id())).isEqualTo("WAITING");
            assertThat(upload(id(), png, "其他文件.png", null, USER).state()).isEqualTo("READY");
        } finally {
            release.countDown();
            try {
                assertThat(writer.get(5, TimeUnit.SECONDS)).isPositive();
            } finally {
                remoteRegistry.destroy();
                remoteWorker.close();
            }
        }
        assertThatThrownBy(() -> registry.finish(claim, 100, USER)).isInstanceOf(MediaException.class);
        jdbc.update("UPDATE afv_media_blob_cleanup SET next_attempt_at=UTC_TIMESTAMP(6) WHERE blob_id=?",
                blob.id());
        assertThat(worker.runOne()).isTrue();
        assertThat(directory.resolve(".afv-managed").resolve(blob.objectKey())).doesNotExist();
        assertThatThrownBy(() -> blobs.write(blob, source)).isInstanceOf(IOException.class)
                .hasMessageContaining("不能重新写入旧对象");
    }

    @Test
    void slowS3UploadDoesNotBlockExpirationOrAllowAnOldGenerationToPublish() throws Exception {
        jdbc.update("UPDATE afv_storage_config SET is_default=(id=72)");
        Path source = directory.resolve("s3-slow-" + id() + ".png");
        Files.write(source, png);
        var facts = context.getBean(MediaProbe.class).inspect(source, "image/png");
        String request = id();
        var claim = registry.admit(request, null, "对象上传.png", facts, USER);
        var blob = repository.blob(claim.blobId());
        var actual = context.getBean(S3ClientFactory.class)
                .getClient(context.getBean(S3StorageConfigResolver.class).resolve(config(72)));
        var client = spy(actual);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        doAnswer(call -> {
            pauseIo(entered, release);
            return actual.putObject(call.getArgument(0, PutObjectRequest.class),
                    call.getArgument(1, RequestBody.class));
        }).when(client)
                .putObject(argThat((PutObjectRequest requestValue) -> requestValue.key().endsWith(blob.objectKey())),
                        any(RequestBody.class));
        var factory = context.getBean(S3ClientFactory.class);
        doReturn(client).when(factory).getClient(any());
        var writer = CompletableFuture.supplyAsync(() -> {
            try {
                return blobs.write(blob, source);
            } catch (IOException failure) {
                throw new CompletionException(failure);
            }
        });
        try {
            assertThat(entered.await(5, TimeUnit.SECONDS)).isTrue();
            CompletableFuture.runAsync(() -> {
                jdbc.update(
                        "UPDATE afv_media_resource SET lease_until=TIMESTAMPADD(MINUTE,-1,UTC_TIMESTAMP(6)) WHERE id=?",
                        claim.resource().id());
                registry.expireUploads();
            }).get(5, TimeUnit.SECONDS);
            var replacement = upload(request, png, "对象上传.png", null, USER);
            assertThat(replacement.state()).isEqualTo("READY");
            assertThat(repository.get(replacement.id(), false).blobId()).isNotEqualTo(blob.id());
        } finally {
            release.countDown();
            writer.get(5, TimeUnit.SECONDS);
        }
        assertThatThrownBy(() -> registry.finish(claim, 100, USER)).isInstanceOf(MediaException.class);
        assertThat(repository.get(claim.resource().id(), false).resource().state()).isEqualTo("READY");
    }

    @Test
    void slowS3DeleteReleasesTheCleanupTransactionBeforeStorageIo() throws Exception {
        jdbc.update("UPDATE afv_storage_config SET is_default=(id=72)");
        var orphan = orphan(true);
        jobs.discover();
        var claim = jobs.claim(id()).orElseThrow();
        var actual = context.getBean(S3ClientFactory.class)
                .getClient(context.getBean(S3StorageConfigResolver.class).resolve(config(72)));
        var client = spy(actual);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        doAnswer(call -> {
            pauseIo(entered, release);
            Consumer<DeleteObjectRequest.Builder> configure = call.getArgument(0);
            var request = DeleteObjectRequest.builder();
            configure.accept(request);
            return actual.deleteObject(request.build());
        }).when(client).deleteObject(any(Consumer.class));
        var factory = context.getBean(S3ClientFactory.class);
        doReturn(client).when(factory).getClient(any());
        var deletion = CompletableFuture.supplyAsync(() -> {
            try {
                return blobs.purge(orphan.blob(), () -> jobs.mayDelete(claim));
            } catch (IOException failure) {
                throw new CompletionException(failure);
            }
        });
        try {
            assertThat(entered.await(5, TimeUnit.SECONDS)).isTrue();
            CompletableFuture.runAsync(() -> jobs.defer(claim, false)).get(5, TimeUnit.SECONDS);
            assertThat(upload(id(), png, "清理期间上传.png", null, USER).state()).isEqualTo("READY");
        } finally {
            release.countDown();
            assertThat(deletion.get(5, TimeUnit.SECONDS)).isTrue();
        }
        jobs.complete(claim);
        assertThat(state(orphan)).isEqualTo("ORPHAN");
        jdbc.update("UPDATE afv_media_blob_cleanup SET next_attempt_at=UTC_TIMESTAMP(6) WHERE blob_id=?",
                orphan.blob().id());
        assertThat(worker.runOne()).isTrue();
        assertThat(state(orphan)).isEqualTo("PURGED");
    }

    @Test
    void lostObjectLockFailsAndCannotReleaseAnotherInstancesLock() throws Exception {
        var orphan = orphan(false);
        var redis = new StringRedisTemplate(context.getBean(RedisConnectionFactory.class));
        String key = "afv:media:blob:MEDIA:" + orphan.blob().id();
        var competing = new RedisLockRegistry(context.getBean(RedisConnectionFactory.class), "afv:media:blob");
        var other = competing.obtain("MEDIA:" + orphan.blob().id());
        try {
            assertThatThrownBy(
                    () -> context.getBean(MediaBlobLocks.class).exclusive(orphan.blob().id(), Domain.MEDIA, state -> {
                        redis.delete(key);
                        assertThat(other.tryLock()).isTrue();
                        return true;
                    })).isInstanceOf(IllegalStateException.class);
            assertThat(redis.hasKey(key)).isTrue();
        } finally {
            other.unlock();
            competing.destroy();
        }
        assertThat(state(orphan)).isEqualTo("ORPHAN");
    }

    @Test
    void callerTransactionIsRejectedBeforeReadingTheUploadOrWritingStorage() throws Exception {
        var source = mock(ByteArrayInputStream.class);
        var transaction = context.getBean(TransactionTemplate.class);
        assertThatThrownBy(() -> transaction.execute(status -> {
            try {
                return intake.upload(source, "错误调用.png", "image/png", id(), null, USER);
            } catch (IOException failure) {
                throw new CompletionException(failure);
            }
        })).isInstanceOf(IllegalTransactionStateException.class);
        var synchronization = new TransactionTemplate(context.getBean(PlatformTransactionManager.class));
        synchronization.setPropagationBehavior(TransactionDefinition.PROPAGATION_SUPPORTS);
        assertThatThrownBy(() -> synchronization.execute(status -> {
            assertThat(TransactionSynchronizationManager.isActualTransactionActive()).isFalse();
            try {
                return intake.upload(source, "空同步作用域.png", "image/png", id(), null, USER);
            } catch (IOException failure) {
                throw new CompletionException(failure);
            }
        })).isInstanceOf(IllegalTransactionStateException.class);
        verifyNoInteractions(source);
        var orphan = orphan(false);
        assertThatThrownBy(() -> transaction.execute(status -> {
            try {
                return blobs.purge(orphan.blob(), () -> true);
            } catch (IOException failure) {
                throw new CompletionException(failure);
            }
        })).isInstanceOf(IllegalTransactionStateException.class);
        assertThat(directory.resolve(".afv-managed").resolve(orphan.blob().objectKey())).exists();
    }

    private void pauseIo(CountDownLatch entered, CountDownLatch release) throws Exception {
        assertThat(TransactionSynchronizationManager.isActualTransactionActive()).isFalse();
        assertThat(TransactionSynchronizationManager.hasResource(jdbc.getDataSource())).isFalse();
        entered.countDown();
        assertThat(release.await(20, TimeUnit.SECONDS)).isTrue();
    }

    @Test
    void s3OrphansArePhysicallyDeletedAndMissingObjectsAreIdempotent() throws Exception {
        jdbc.update("UPDATE afv_storage_config SET is_default=(id=72)");
        var orphan = orphan(true);
        var client = context.getBean(S3ClientFactory.class)
                .getClient(context.getBean(S3StorageConfigResolver.class).resolve(config(72)));
        String key = "test/.afv-managed/" + orphan.blob().objectKey();
        assertThat(client.headObject(request -> request.bucket("media-test").key(key)).contentLength()).isPositive();
        jobs.discover();
        assertThat(worker.runOne()).isTrue();
        assertThat(state(orphan)).isEqualTo("PURGED");
        assertThatThrownBy(
                () -> client.headObject(request -> request.bucket("media-test").key(key))).isInstanceOfSatisfying(
                S3Exception.class, failure -> assertThat(failure.statusCode()).isEqualTo(404));
        assertThat(blobs.purge(orphan.blob(), () -> true)).isTrue();
    }

    private Orphan orphan(boolean aged) throws Exception {
        Path source = directory.resolve("intake-" + id() + ".png");
        Files.write(source, png);
        var facts = context.getBean(MediaProbe.class).inspect(source, "image/png");
        var claim = registry.admit(id(), null, "失败接收.png", facts, USER);
        var blob = repository.blob(claim.blobId());
        blobs.write(blob, source);
        registry.fail(claim);
        if (aged) {
            jdbc.update("UPDATE afv_media_blob SET orphaned_at=TIMESTAMPADD(DAY,-31,UTC_TIMESTAMP(6)) WHERE id=?",
                    blob.id());
        }
        return new Orphan(claim.resource(), blob, source);
    }

    private String state(Orphan orphan) {
        return jdbc.queryForObject("SELECT state FROM afv_media_blob WHERE id=?", String.class, orphan.blob().id());
    }
}
