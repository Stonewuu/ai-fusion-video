package com.stonewu.fusion.service.storyboard;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
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
                scriptMapper
        );
    }

    @Test
    void getByProjectIdSelectsOnlyTheLatestEffectiveStoryboard() {
        Script script = Script.builder().id(7L).projectId(3L).build();
        Storyboard expected = Storyboard.builder().id(9L).projectId(3L).build();
        when(scriptMapper.selectPage(any(Page.class), any())).thenReturn(new Page<Script>().setRecords(List.of(script)));
        when(storyboardMapper.selectPage(any(Page.class), any())).thenReturn(new Page<Storyboard>().setRecords(List.of(expected)));

        Storyboard actual = storyboardService.getByProjectId(3L);

        ArgumentCaptor<LambdaQueryWrapper<Storyboard>> queryCaptor = ArgumentCaptor.forClass(
                LambdaQueryWrapper.class);
        ArgumentCaptor<Page<Storyboard>> pageCaptor = ArgumentCaptor.forClass(Page.class);
        verify(storyboardMapper).selectPage(pageCaptor.capture(), queryCaptor.capture());
        assertThat(actual).isSameAs(expected);
        assertThat(queryCaptor.getValue().getSqlSegment().toUpperCase())
                .contains("ORDER BY");
        assertThat(pageCaptor.getValue().getSize()).isEqualTo(1);
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
}
