package com.stonewu.fusion.integration;

import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import com.stonewu.fusion.mapper.execution.ExecutionRecordMapper;
import com.stonewu.fusion.mapper.execution.ExecutionRecoveryDecisionMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.service.ai.run.AgentEventEnvelopeSanitizer;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.execution.ExecutionRecordQueries;
import com.stonewu.fusion.service.execution.ExecutionRecordGenerationDetails;
import com.stonewu.fusion.service.execution.ExecutionRecordQueries.Filter;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Kind;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Ref;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Remote;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Target;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryTargets;
import com.zaxxer.hikari.HikariDataSource;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 筛选必须先于分页，并且列表、统计与详情都按原调用用户隔离。 */
@Testcontainers
class ExecutionRecordFilteringIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("execution_records").withUsername("afv").withPassword("afv-test");

    @Test
    void filtersOlderExceptionsBeforePaginationWithoutCountingAnotherUsersRecords() throws Exception {
        try (var source = new HikariDataSource(); var context = new AnnotationConfigApplicationContext()) {
            source.setJdbcUrl(MYSQL.getJdbcUrl());
            source.setUsername(MYSQL.getUsername());
            source.setPassword(MYSQL.getPassword());
            var flyway = Flyway.configure().dataSource(source).load();
            flyway.migrate();
            MybatisTestSupport.register(context, source, ExecutionRecordMapper.class,
                    GenerationSubmissionMapper.class, RemoteGenerationJobMapper.class);
            context.refresh();
            var submissions = context.getBean(GenerationSubmissionMapper.class);
            var jobs = context.getBean(RemoteGenerationJobMapper.class);
            var records = context.getBean(ExecutionRecordMapper.class);
            var now = LocalDateTime.now(ZoneOffset.UTC);
            for (int index = 0; index < 24; index++) {
                insert(submissions, jobs, 1L, index + 1, "FAILED", now.minusMinutes(index));
            }
            String review = insert(submissions, jobs, 1L, 25, "UNKNOWN_SUBMISSION", now.minusDays(2));
            String cancelled = insert(submissions, jobs, 1L, 26, "CANCELLED", now.minusDays(10));
            insert(submissions, jobs, 1L, 27, "FAILED", now.minusDays(35));
            insert(submissions, jobs, 1L, 28, "SUCCEEDED", now);
            insert(submissions, jobs, 1L, 29, "RUNNING", now);
            String otherUser = insert(submissions, jobs, 2L, 30, "UNKNOWN_SUBMISSION", now);
            var targets = mock(ExecutionRecoveryTargets.class);
            when(targets.get(any())).thenAnswer(call -> {
                Ref ref = call.getArgument(0);
                var job = jobs.selectById(ref.id());
                return new Target(ref.kind(), ref.id(), job.getState(), 1,
                        new CanvasActor(job.getActorUserId(), null), null, null, null, false, false,
                        List.of(new Remote("生成", "attempt", null, false, false)));
            });
            var queries = new ExecutionRecordQueries(records, targets, mock(ExecutionRecoveryDecisionMapper.class),
                    mock(AgentEventEnvelopeSanitizer.class), mock(ExecutionRecordGenerationDetails.class));

            var first = queries.list(1L, 0, Filter.ALL, 0);
            assertThat(first.total()).isEqualTo(27);
            assertThat(first.items()).hasSize(20).allSatisfy(row -> assertThat(row.status()).isEqualTo("FAILED"));
            assertThat(first.hasMore()).isTrue();
            var second = queries.list(1L, 1, Filter.ALL, 0);
            assertThat(second.items()).hasSize(7);
            assertThat(second.hasMore()).isFalse();
            var awaiting = queries.list(1L, 0, Filter.REVIEW, 0);
            assertThat(awaiting.items()).extracting(ExecutionRecordQueries.Summary::id).containsExactly(review);
            assertThat(awaiting.total()).isEqualTo(27);
            assertThat(awaiting.matchedTotal()).isEqualTo(1);
            assertThat(queries.list(1L, 0, Filter.ALL, 7).matchedTotal()).isEqualTo(25);
            assertThat(queries.list(1L, 0, Filter.ALL, 30).matchedTotal()).isEqualTo(26);
            assertThat(queries.list(1L, 0, Filter.CANCELLED, 0).items())
                    .extracting(ExecutionRecordQueries.Summary::id).containsExactly(cancelled);
            assertThat(queries.list(1L, 0, Filter.CANCELLED, 7).items()).isEmpty();
            assertThat(records.get(1L, Kind.GENERATION, otherUser)).isNull();
            assertThat(queries.list(2L, 0, Filter.ALL, 0).total()).isEqualTo(1);
            assertThatThrownBy(() -> queries.list(1L, 0, Filter.ALL, 3)).hasMessageContaining("请选择");
        }
    }

    private String insert(GenerationSubmissionMapper submissions, RemoteGenerationJobMapper jobs,
                          long user, long item, String status, LocalDateTime time) {
        String submissionId = UUID.randomUUID().toString();
        submissions.insert(GenerationSubmissionEntity.builder().id(submissionId).requestId(submissionId)
                .scopeKey("s".repeat(64)).requestHash("a".repeat(64)).kind("IMAGE").taskId(item)
                .publicTaskId(UUID.randomUUID().toString()).actorUserId(user).contextJson("{\"origin\":\"workbench\"}")
                .modelSnapshotJson("{\"code\":\"test-image\"}").apiConfigId(1L).apiConfigHash("a".repeat(64))
                .state("FAILED").createdAt(time).updatedAt(time).build());
        String id = UUID.randomUUID().toString();
        jobs.insert(RemoteGenerationJobEntity.builder().id(id).submissionId(submissionId).itemId(item).itemIndex(0)
                .modelId(1L).actorUserId(user).state(status).phase("FINISHED").inputJson("{}")
                .cancelRequested(false).cancelSent(false).permitHeld(false).nextPollAt(time).deadlineAt(time.plusHours(2))
                .createdAt(time).updatedAt(time).build());
        return id;
    }
}
