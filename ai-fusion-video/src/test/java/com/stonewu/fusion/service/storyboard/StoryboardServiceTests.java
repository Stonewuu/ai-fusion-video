package com.stonewu.fusion.service.storyboard;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardSceneMapper;
import com.stonewu.fusion.service.storyboard.dto.StoryboardItemAssetsPatch;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class StoryboardServiceTests {

    @BeforeAll
    static void initializeMybatisTableMetadata() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(new MybatisConfiguration(), "");
        TableInfoHelper.initTableInfo(assistant, Script.class);
        TableInfoHelper.initTableInfo(assistant, Storyboard.class);
        TableInfoHelper.initTableInfo(assistant, StoryboardEpisode.class);
        TableInfoHelper.initTableInfo(assistant, StoryboardScene.class);
        TableInfoHelper.initTableInfo(assistant, StoryboardItem.class);
    }

    @Mock
    private StoryboardMapper storyboardMapper;

    @Mock
    private StoryboardEpisodeMapper episodeMapper;

    @Mock
    private StoryboardSceneMapper sceneMapper;

    @Mock
    private StoryboardItemMapper itemMapper;

    @Mock
    private ScriptMapper scriptMapper;

    @Mock
    private ScriptEpisodeMapper scriptEpisodeMapper;

    private StoryboardService storyboardService;

    @BeforeEach
    void setUp() {
        storyboardService = new StoryboardService(
                storyboardMapper,
                episodeMapper,
                sceneMapper,
                itemMapper,
                scriptMapper,
                scriptEpisodeMapper
        );
    }

    @Test
    void getByProjectIdSelectsOnlyTheLatestEffectiveStoryboard() {
        Script script = Script.builder().id(7L).projectId(3L).build();
        Storyboard expected = Storyboard.builder().id(9L).projectId(3L).build();
        when(scriptMapper.selectOne(any())).thenReturn(script);
        when(storyboardMapper.selectOne(any())).thenReturn(expected);

        Storyboard actual = storyboardService.getByProjectId(3L);

        ArgumentCaptor<LambdaQueryWrapper<Storyboard>> queryCaptor = ArgumentCaptor.forClass(
                LambdaQueryWrapper.class);
        verify(storyboardMapper).selectOne(queryCaptor.capture());
        assertThat(actual).isSameAs(expected);
        assertThat(queryCaptor.getValue().getSqlSegment().toUpperCase())
                .contains("ORDER BY", "LIMIT 1");
    }

    @Test
    void updateCannotChangeProjectScriptIdentityOrFixedTitle() {
        Storyboard existing = Storyboard.builder()
                .id(9L)
                .projectId(3L)
                .scriptId(7L)
                .title("固定项目名")
                .ownerType(2)
                .ownerId(88L)
                .description("旧描述")
                .build();
        when(storyboardMapper.selectById(9L)).thenReturn(existing);
        Storyboard update = Storyboard.builder()
                .id(9L)
                .projectId(999L)
                .scriptId(777L)
                .title("不允许修改")
                .ownerType(1)
                .ownerId(1L)
                .description("新描述")
                .build();

        Storyboard actual = storyboardService.update(update);

        assertThat(actual.getProjectId()).isEqualTo(3L);
        assertThat(actual.getScriptId()).isEqualTo(7L);
        assertThat(actual.getTitle()).isEqualTo("固定项目名");
        assertThat(actual.getOwnerType()).isEqualTo(2);
        assertThat(actual.getOwnerId()).isEqualTo(88L);
        assertThat(actual.getDescription()).isEqualTo("新描述");
    }

    @Test
    void bindScriptEpisodeRejectsCrossScriptEpisode() {
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(21L)
                .build());
        when(storyboardMapper.selectById(21L)).thenReturn(Storyboard.builder()
                .id(21L)
                .scriptId(31L)
                .build());
        when(scriptEpisodeMapper.selectById(41L)).thenReturn(ScriptEpisode.builder()
                .id(41L)
                .scriptId(99L)
                .build());

        assertThatThrownBy(() -> storyboardService.bindScriptEpisode(11L, 41L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("剧本分集不属于当前分镜关联的剧本");

        verify(episodeMapper, never()).updateById(any(StoryboardEpisode.class));
    }

    @Test
    void saveEpisodeForScriptReusesExistingBinding() {
        StoryboardEpisode existing = StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(21L)
                .scriptEpisodeId(41L)
                .episodeNumber(1)
                .build();
        StoryboardEpisode updated = StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(21L)
                .scriptEpisodeId(41L)
                .episodeNumber(1)
                .title("第一集")
                .build();

        when(storyboardMapper.selectById(21L)).thenReturn(Storyboard.builder()
                .id(21L)
                .scriptId(31L)
                .build());
        when(scriptEpisodeMapper.selectById(41L)).thenReturn(ScriptEpisode.builder()
                .id(41L)
                .scriptId(31L)
                .episodeNumber(1)
                .title("第一集")
                .build());
        when(episodeMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(existing);
        when(episodeMapper.selectById(11L)).thenReturn(updated);

        StoryboardEpisode result = storyboardService.saveEpisodeForScript(
                21L, 41L, 1, "第一集", "梗概");

        assertThat(result.getId()).isEqualTo(11L);
        verify(episodeMapper).updateById(any(StoryboardEpisode.class));
        verify(episodeMapper, never()).insert(any(StoryboardEpisode.class));
    }

    @Test
    void clearEpisodeContentDeletesOnlyTargetEpisodeScenesAndItems() {
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(21L)
                .build());
        when(sceneMapper.selectList(any(LambdaQueryWrapper.class))).thenReturn(List.of(
                StoryboardScene.builder().id(101L).episodeId(11L).build(),
                StoryboardScene.builder().id(102L).episodeId(11L).build()
        ));

        storyboardService.clearEpisodeContent(11L);

        verify(itemMapper).delete(any(LambdaQueryWrapper.class));
        verify(sceneMapper).delete(any(LambdaQueryWrapper.class));
    }

    @Test
    void createSceneWithItemsDerivesAndPersistsAllParentIds() {
        StoryboardScene scene = StoryboardScene.builder()
                .storyboardId(21L)
                .episodeId(11L)
                .build();
        StoryboardItem item = StoryboardItem.builder().content("镜头").build();
        when(storyboardMapper.selectById(21L)).thenReturn(Storyboard.builder().id(21L).build());
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(21L)
                .build());
        when(sceneMapper.insert(scene)).thenAnswer(invocation -> {
            scene.setId(31L);
            return 1;
        });

        StoryboardScene result = storyboardService.createSceneWithItems(scene, List.of(item));

        assertThat(result.getId()).isEqualTo(31L);
        assertThat(item.getStoryboardId()).isEqualTo(21L);
        assertThat(item.getStoryboardEpisodeId()).isEqualTo(11L);
        assertThat(item.getStoryboardSceneId()).isEqualTo(31L);
        verify(itemMapper).insert(item);
    }

    @Test
    void createSceneWithItemsRejectsEpisodeFromAnotherStoryboard() {
        StoryboardScene scene = StoryboardScene.builder()
                .storyboardId(21L)
                .episodeId(11L)
                .build();
        when(storyboardMapper.selectById(21L)).thenReturn(Storyboard.builder().id(21L).build());
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder()
                .id(11L)
                .storyboardId(22L)
                .build());

        assertThatThrownBy(() -> storyboardService.createSceneWithItems(scene, List.of()))
                .isInstanceOf(BusinessException.class)
                .hasMessage("分镜集不属于指定分镜");

        verify(sceneMapper, never()).insert(any(StoryboardScene.class));
    }

    @Test
    void deleteSceneAlsoDeletesItsItems() {
        when(sceneMapper.selectById(31L)).thenReturn(StoryboardScene.builder().id(31L).build());

        storyboardService.deleteScene(31L);

        verify(itemMapper).delete(any(LambdaQueryWrapper.class));
        verify(sceneMapper).deleteById(31L);
    }

    @Test
    void getStatisticsUsesCountsInsteadOfLoadingStoryboardDetails() {
        when(episodeMapper.selectCount(any(LambdaQueryWrapper.class))).thenReturn(4L);
        when(sceneMapper.selectCount(any(LambdaQueryWrapper.class))).thenReturn(10L);
        when(itemMapper.selectCount(any(LambdaQueryWrapper.class))).thenReturn(36L);

        StoryboardStatistics statistics = storyboardService.getStatistics(21L);

        assertThat(statistics).isEqualTo(new StoryboardStatistics(4L, 10L, 36L));
        verify(episodeMapper, never()).selectList(any(LambdaQueryWrapper.class));
        verify(sceneMapper, never()).selectList(any(LambdaQueryWrapper.class));
        verify(itemMapper, never()).selectList(any(LambdaQueryWrapper.class));
    }

    @Test
    @SuppressWarnings("unchecked")
    void updateItemAssetsClearsCharacterPropAndSceneAssociations() {
        StoryboardItem existing = StoryboardItem.builder()
                .id(51L)
                .characterIds("[1]")
                .sceneAssetItemId(2L)
                .propIds("[3]")
                .build();
        StoryboardItem cleared = StoryboardItem.builder()
                .id(51L)
                .characterIds("[]")
                .propIds("[]")
                .build();
        when(itemMapper.selectById(51L)).thenReturn(existing, cleared);

        StoryboardItem result = storyboardService.updateItemAssets(51L, new StoryboardItemAssetsPatch(
                true, List.of(),
                true, null,
                true, List.of()
        ));

        ArgumentCaptor<UpdateWrapper<StoryboardItem>> updateCaptor =
                ArgumentCaptor.forClass(UpdateWrapper.class);
        verify(itemMapper).update(isNull(), updateCaptor.capture());
        UpdateWrapper<StoryboardItem> update = updateCaptor.getValue();
        String sqlSet = update.getSqlSet();
        assertThat(sqlSet)
                .contains("character_ids", "scene_asset_item_id", "prop_ids");
        assertThat(update.getParamNameValuePairs().values())
                .contains("[]")
                .containsNull();
        assertThat(result).isSameAs(cleared);
    }

    @Test
    void updateItemAssetsWithNoPresentFieldsDoesNotIssueUpdate() {
        StoryboardItem existing = StoryboardItem.builder().id(51L).build();
        when(itemMapper.selectById(51L)).thenReturn(existing);

        StoryboardItem result = storyboardService.updateItemAssets(51L, new StoryboardItemAssetsPatch(
                false, null,
                false, null,
                false, null
        ));

        verify(itemMapper, never()).update(any(), any());
        assertThat(result).isSameAs(existing);
    }

    @Test
    @SuppressWarnings("unchecked")
    void updateItemAssetsDoesNotTouchMissingFields() {
        StoryboardItem existing = StoryboardItem.builder().id(51L).build();
        StoryboardItem updated = StoryboardItem.builder()
                .id(51L)
                .characterIds("[7,8]")
                .build();
        when(itemMapper.selectById(51L)).thenReturn(existing, updated);

        storyboardService.updateItemAssets(51L, new StoryboardItemAssetsPatch(
                true, List.of(7L, 8L),
                false, null,
                false, null
        ));

        ArgumentCaptor<UpdateWrapper<StoryboardItem>> updateCaptor =
                ArgumentCaptor.forClass(UpdateWrapper.class);
        verify(itemMapper).update(isNull(), updateCaptor.capture());
        UpdateWrapper<StoryboardItem> update = updateCaptor.getValue();
        String sqlSet = update.getSqlSet();
        assertThat(sqlSet)
                .contains("character_ids")
                .doesNotContain("scene_asset_item_id", "prop_ids");
        assertThat(update.getParamNameValuePairs().values()).contains("[7,8]");
    }
}
