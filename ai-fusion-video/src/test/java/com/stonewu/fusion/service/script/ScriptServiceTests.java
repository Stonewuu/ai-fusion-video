package com.stonewu.fusion.service.script;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ScriptServiceTests {

    @BeforeAll
    static void initializeMybatisTableMetadata() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(new MybatisConfiguration(), "");
        TableInfoHelper.initTableInfo(assistant, Script.class);
        TableInfoHelper.initTableInfo(assistant, ScriptEpisode.class);
        TableInfoHelper.initTableInfo(assistant, ScriptSceneItem.class);
    }

    @Mock
    private ScriptMapper scriptMapper;

    @Mock
    private ScriptEpisodeMapper episodeMapper;

    @Mock
    private ScriptSceneItemMapper sceneItemMapper;

    @InjectMocks
    private ScriptService scriptService;

    @Test
    void getByProjectIdSelectsOnlyTheLatestEffectiveScript() {
        Script expected = Script.builder().id(9L).projectId(3L).build();
        when(scriptMapper.selectOne(any())).thenReturn(expected);

        Script actual = scriptService.getByProjectId(3L);

        ArgumentCaptor<LambdaQueryWrapper<Script>> queryCaptor = ArgumentCaptor.forClass(
                LambdaQueryWrapper.class);
        verify(scriptMapper).selectOne(queryCaptor.capture());
        assertThat(actual).isSameAs(expected);
        assertThat(queryCaptor.getValue().getSqlSegment().toUpperCase())
                .contains("ORDER BY", "LIMIT 1");
    }

    @Test
    void updateCannotChangeProjectIdentityOrFixedTitle() {
        Script existing = Script.builder()
                .id(9L)
                .projectId(3L)
                .title("固定项目名")
                .ownerType(2)
                .ownerId(88L)
                .content("旧内容")
                .build();
        when(scriptMapper.selectById(9L)).thenReturn(existing);
        when(scriptMapper.updateById(any(Script.class))).thenReturn(1);
        Script update = Script.builder()
                .id(9L)
                .projectId(999L)
                .title("不允许修改")
                .ownerType(1)
                .ownerId(1L)
                .content("新内容")
                .build();

        Script actual = scriptService.update(update);

        assertThat(actual.getProjectId()).isEqualTo(3L);
        assertThat(actual.getTitle()).isEqualTo("固定项目名");
        assertThat(actual.getOwnerType()).isEqualTo(2);
        assertThat(actual.getOwnerId()).isEqualTo(88L);
        assertThat(actual.getContent()).isEqualTo("新内容");
    }

    @Test
    void deleteEpisodeAlsoDeletesItsScenes() {
        when(episodeMapper.selectById(17L)).thenReturn(ScriptEpisode.builder().id(17L).build());

        scriptService.deleteEpisode(17L);

        verify(sceneItemMapper).delete(any(LambdaQueryWrapper.class));
        verify(episodeMapper).deleteById(17L);
    }
}
