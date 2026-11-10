package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourceDeleteCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.StoryboardChildrenCommand;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 验证统一写入事务不会留下漏绑镜头，失败时场次、镜头和父版本一起回滚。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@Testcontainers(disabledWithoutDocker = false)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class StoryboardSceneBindingIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);

    @DynamicPropertySource
    static void configure(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        properties.add("spring.data.redis.password", () -> "");
        properties.add("fusion.agentscope.v2.state.mode", () -> "in-memory");
        properties.add("fusion.agentscope.v2.execution.maintenance-initial-delay-ms", () -> 3600000);
    }

    @Autowired ResourceCommandService commands;
    @Autowired ResourceQueryService resources;
    @Autowired StoryboardResourceQueryService boards;
    @Autowired UserMapper users;
    @Autowired CanvasJson json;
    CanvasActor actor;
    ResourceCommandContext user;
    long project;
    long board;
    long episode;
    ObjectNode sceneAsset;
    long sceneItem;

    @BeforeEach
    void setup() {
        var account = User.builder().username("scene-test-" + UUID.randomUUID()).password("test").build();
        users.insert(account);
        actor = new CanvasActor(account.getId(), null);
        user = ResourceCommandContext.user(actor, null);
        project = create(PROJECT, null, Map.of("name", "场景绑定验证")).path("id").longValue();
        board = boards.project(project, actor).path("id").longValue();
        episode = create(STORYBOARD_EPISODE, project,
                Map.of("storyboardId", board, "parentVersion", 0, "episodeNumber", 1)).path("id").longValue();
        sceneAsset = create(ASSET, project, Map.of("name", "雨中道路", "type", "scene"));
        sceneItem = sceneAsset.path("items").get(0).path("id").longValue();
    }

    @Test
    void missingSceneRollsBackTheEntireSceneAndParentVersion() {
        var draft = json.object(Map.of("sceneNumber", "1-1", "shots", List.of(
                Map.of("content", "山路空镜", "sceneAssetItemId", sceneItem), Map.of("content", "人物走入"))));
        assertThatThrownBy(() -> append(episode, 0, draft)).isInstanceOf(CanvasException.class)
                .hasMessageContaining("关联场景");
        assertThat(boards.children(STORYBOARD_SCENE, STORYBOARD_EPISODE, episode, actor)).isEmpty();
        assertThat(resources.get(STORYBOARD_EPISODE, episode, actor).header().version()).isZero();
        assertThat(boards.statistics(board, actor).itemCount()).isZero();
    }

    @Test
    void normalSaveBindsEmptyShotsAndCloseupsAndRejectsClearingScene() {
        var shots = saveShots();
        assertThat(shots).hasSize(2).allSatisfy(shot -> assertThat(shot.path("sceneAssetItemId").longValue()).isEqualTo(sceneItem));
        var first = shots.get(0);
        assertThatThrownBy(() -> commands.execute(new ResourcePatchCommand(key(), SHOT, first.path("id").longValue(),
                first.path("version").intValue(), json.object(Map.of()).putNull("sceneAssetItemId")), user))
                .isInstanceOf(CanvasException.class).hasMessageContaining("关联场景");
        var saved = resources.get(SHOT, first.path("id").longValue(), actor);
        assertThat(saved.data().path("sceneAssetItemId").longValue()).isEqualTo(sceneItem);
        assertThat(saved.header().version()).isEqualTo(first.path("version").intValue());
        assertThat(boards.shotCounts(board, actor)).singleElement().satisfies(count -> {
            assertThat(count.itemCount()).isEqualTo(2);
            assertThat(count.missingSceneCount()).isZero();
        });
    }

    @Test
    void sceneMustBelongToTheProjectAndHaveSceneType() {
        long otherProject = create(PROJECT, null, Map.of("name", "另一项目")).path("id").longValue();
        long otherScene = create(ASSET, otherProject, Map.of("name", "道路", "type", "scene"))
                .path("items").get(0).path("id").longValue();
        long person = create(ASSET, project, Map.of("name", "人物", "type", "character"))
                .path("items").get(0).path("id").longValue();
        for (long invalidScene : List.of(otherScene, person)) {
            assertThatThrownBy(() -> append(episode, 0, json.object(Map.of("shots", List.of(
                    Map.of("content", "道路空镜", "sceneAssetItemId", invalidScene))))))
                    .isInstanceOf(CanvasException.class);
        }
        assertThat(boards.statistics(board, actor).itemCount()).isZero();
    }

    @Test
    void inUseSceneCannotBeDeletedButClearingStoryboardReleasesIt() {
        saveShots();
        long assetId = sceneAsset.path("id").longValue();
        assertThatThrownBy(() -> commands.execute(new ResourceDeleteCommand(key(), ASSET, assetId, 0), user))
                .isInstanceOf(CanvasException.class);
        assertThat(resources.get(ASSET_ITEM, sceneItem, actor).data().path("assetId").longValue()).isEqualTo(assetId);
        clearBoard();
        commands.execute(new ResourceDeleteCommand(key(), ASSET, assetId, 0), user);
        assertThat(boards.statistics(board, actor).itemCount()).isZero();
    }

    @Test
    void deletingWholeProjectDoesNotLeaveActiveSceneReferences() {
        saveShots();
        int version = resources.get(PROJECT, project, actor).header().version();
        commands.execute(new ResourceDeleteCommand(key(), PROJECT, project, version), user);
        assertThatThrownBy(() -> resources.get(STORYBOARD, board, actor)).isInstanceOf(CanvasException.class);
    }

    private List<ObjectNode> saveShots() {
        append(episode, 0, json.object(Map.of("sceneNumber", "1-1", "shots", List.of(
                Map.of("content", "山路空镜", "sceneAssetItemId", sceneItem),
                Map.of("content", "脚步特写", "sceneAssetItemId", sceneItem)))));
        return boards.children(SHOT, STORYBOARD_EPISODE, episode, actor);
    }

    private void clearBoard() {
        commands.execute(new StoryboardChildrenCommand(key(), STORYBOARD, board,
                resources.get(STORYBOARD, board, actor).header().version(), StoryboardChildrenCommand.Mode.CLEAR,
                List.of(), List.of()), user);
    }

    private void append(long parent, int version, ObjectNode child) {
        commands.execute(new StoryboardChildrenCommand(key(), STORYBOARD_EPISODE, parent, version,
                StoryboardChildrenCommand.Mode.APPEND, List.of(child), List.of()), user);
    }

    private ObjectNode create(CanvasValueType type, Long projectId, Map<String, ?> fields) {
        return commands.execute(new ResourceCreateCommand(key(), type, projectId, json.object(fields)), user).result();
    }

    private String key() {
        return UUID.randomUUID().toString();
    }
}
