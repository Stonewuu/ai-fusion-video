package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.stonewu.fusion.mapper.generation.AudioTaskMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.stonewu.fusion.service.generation.audio.AudioTaskData;
import com.stonewu.fusion.service.generation.audio.AudioTaskRepository;
import com.stonewu.fusion.service.ai.google.GoogleAudioProtocol;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.zaxxer.hikari.HikariDataSource;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 在隔离 MySQL 中验证并发额度、失效工作者隔离、结果保存重试和用户作用域。 */
@Testcontainers
class AudioTaskPersistenceIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("audio_generation_tasks").withUsername("afv").withPassword("afv-test");

    @Configuration
    @EnableTransactionManagement
    static class Transactions {
    }

    @Test
    void receivedOutputCanRetrySavingWithoutResubmittingAndExpiredWorkersCannotPublish() throws Exception {
        try (var source = new HikariDataSource(); var context = new AnnotationConfigApplicationContext()) {
            source.setJdbcUrl(MYSQL.getJdbcUrl());
            source.setUsername(MYSQL.getUsername());
            source.setPassword(MYSQL.getPassword());
            source.setMaximumPoolSize(2);
            var flyway = Flyway.configure().dataSource(source).load();
            flyway.migrate();
            flyway.validate();
            MybatisTestSupport.register(context, source, AudioTaskMapper.class, ExecutionCoordinationMapper.class,
                    AiModelMapper.class, ApiConfigMapper.class);
            context.register(Transactions.class);
            context.registerBean(PlatformTransactionManager.class, () -> new DataSourceTransactionManager(source));
            context.registerBean(CanvasJson.class, () -> new CanvasJson(new ObjectMapper().findAndRegisterModules()));
            context.registerBean(AudioTaskRepository.class);
            context.refresh();
            var json = context.getBean(CanvasJson.class);
            var mapper = context.getBean(AudioTaskMapper.class);
            var clock = context.getBean(ExecutionCoordinationMapper.class);
            var repository = context.getBean(AudioTaskRepository.class);
            var configs = context.getBean(ApiConfigMapper.class);
            var api = ApiConfig.builder().name("目录接口回归").platform("vertex_ai").appId("test-project")
                    .apiUrl("global").build();
            configs.insert(api);
            var models = context.getBean(AiModelMapper.class);
            var model = AiModel.builder().name("语音生成回归").code("gemini-3.8-flash-tts").modelType(4)
                    .apiConfigId(api.getId()).modelProtocol("vertex_ai").maxConcurrency(1).status(1).build();
            models.insert(model);
            var actor = new CanvasActor(1, null);
            var first = task(model, json);
            var second = task(model, json);
            repository.insert(first);
            repository.insert(second);
            var submit = repository.claim("submit").orElseThrow();
            assertThat(submit.getId()).isEqualTo(first.getId());
            assertThat(repository.claim("excess")).isEmpty();
            repository.received(submit, new AudioTaskData.Received("normalized",
                    json.write(new GoogleAudioProtocol.Output("已生成的语音",
                            List.of(new GoogleAudioProtocol.Audio("YQ==".repeat(100000), "audio/wav"))))));
            assertThat(repository.recent(actor)).hasSize(2);
            mapper.update(null, Wrappers.<AudioTask>lambdaUpdate().eq(AudioTask::getId, second.getId())
                    .set(AudioTask::getNextPollAt, clock.now().minusSeconds(1)));
            var next = repository.claim("next").orElseThrow();
            assertThat(next.getId()).isEqualTo(second.getId());
            repository.finish(next, "FAILED", null, "测试结束");
            var save = repository.claim("old-save").orElseThrow();
            assertThat(save.getOutputJson()).contains("已生成的语音");
            mapper.update(null, Wrappers.<AudioTask>lambdaUpdate().eq(AudioTask::getId, first.getId())
                    .set(AudioTask::getLeaseUntil, clock.now().minusSeconds(1)));
            var recovered = repository.claim("new-save").orElseThrow();
            repository.finish(save, "COMPLETED", new AudioTaskData.Result("过期输出", List.of()), null);
            assertThat(repository.owned(first.getId(), actor).getState()).isEqualTo("RUNNING");
            repository.finish(recovered, "FAILED", null, "存储暂时不可用");
            assertThat(repository.owned(first.getId(), actor).getOutputJson()).isNotNull();
            assertThatThrownBy(() -> repository.retrySave(first.getId(), new CanvasActor(2, null)))
                    .hasMessageContaining("不存在");
            repository.retrySave(first.getId(), actor);
            var retried = repository.claim("retry-save").orElseThrow();
            assertThat(retried.getOutputJson()).contains("已生成的语音");
            repository.finish(retried, "COMPLETED", new AudioTaskData.Result("已生成的语音", List.of()), null);
            var finished = repository.owned(first.getId(), actor);
            assertThat(finished.getOutputJson()).isNull();
            assertThat(finished.getResultJson()).contains("已生成的语音");
            assertThatThrownBy(() -> repository.owned(first.getId(), new CanvasActor(1, 7L))).hasMessageContaining("不存在");

            var uncertain = task(model, json);
            repository.insert(uncertain);
            repository.claim("lost-submit").orElseThrow();
            mapper.update(null, Wrappers.<AudioTask>lambdaUpdate().eq(AudioTask::getId, uncertain.getId())
                    .set(AudioTask::getLeaseUntil, clock.now().minusSeconds(1)));
            assertThat(repository.claim("must-not-resubmit")).isEmpty();
            assertThat(repository.owned(uncertain.getId(), actor).getState()).isEqualTo("UNKNOWN");
        }
    }

    private AudioTask task(AiModel model, CanvasJson json) {
        var task = new AudioTask();
        task.setId(UUID.randomUUID().toString());
        task.setRequestId(UUID.randomUUID().toString());
        task.setUserId(1L);
        task.setModelId(model.getId());
        task.setModelName(model.getName());
        task.setModelSnapshotJson(json.write(model));
        task.setApiConfigId(model.getApiConfigId());
        task.setApiRouteJson(json.write(new AudioTaskData.Route("vertex_ai", "global", "test-project")));
        task.setInputJson(json.write(new AudioTaskData.Input("你好，欢迎试听", List.of(), null)));
        task.setInputHash("a".repeat(64));
        task.setState("QUEUED");
        return task;
    }
}
