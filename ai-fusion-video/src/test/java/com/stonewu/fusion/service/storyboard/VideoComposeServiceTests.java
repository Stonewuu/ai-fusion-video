package com.stonewu.fusion.service.storyboard;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.task.TaskStreamService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class VideoComposeServiceTests {

    @Mock
    private StoryboardService storyboardService;

    @Mock
    private StoryboardEpisodeMapper episodeMapper;

    @Mock
    private MediaStorageService mediaStorageService;

    @Mock
    private StorageConfigService storageConfigService;

    @Mock
    private TaskStreamService taskStreamService;

    @Mock
    private Executor videoComposeExecutor;

    private VideoComposeService videoComposeService;

    @BeforeEach
    void setUp() {
        videoComposeService = new VideoComposeService(
                storyboardService,
                episodeMapper,
                mediaStorageService,
                storageConfigService,
                taskStreamService,
                videoComposeExecutor
        );
        ReflectionTestUtils.setField(videoComposeService, "mediaLocalPath", "D:/media-root");
        ReflectionTestUtils.setField(videoComposeService, "allowedHostsConfig", "");
        ReflectionTestUtils.setField(videoComposeService, "maxRedirects", 3);
        ReflectionTestUtils.setField(videoComposeService, "ffmpegPath", "ffmpeg");
        ReflectionTestUtils.setField(videoComposeService, "ffprobePath", "ffprobe");
    }

    @Test
    void submitComposeUsesConditionalUpdateBeforeScheduling() {
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder().id(11L).storyboardId(21L).build());
        when(storyboardService.getById(21L)).thenReturn(Storyboard.builder().id(21L).projectId(31L).build());
        when(taskStreamService.createTask(eq(99L), eq(31L), eq("storyboard_episode_compose"), any(String.class), eq("storyboard_episode"), eq(11L), any(String.class)))
            .thenReturn("task-1");
        when(storyboardService.listScenesByEpisode(11L)).thenReturn(List.of());
        String taskId = videoComposeService.submitCompose(11L, 99L);

        assertThat(taskId).isEqualTo("task-1");
        verify(taskStreamService).fail("task-1", "本集没有可合成的视频，请先生成镜头视频");
        verifyNoInteractions(videoComposeExecutor);
    }

    @Test
    void submitComposeUsesConditionalUpdateBeforeSchedulingWhenVideoExists() {
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder()
            .id(11L)
            .storyboardId(21L)
            .episodeNumber(1)
            .title("第一集")
            .build());
        when(storyboardService.getById(21L)).thenReturn(Storyboard.builder().id(21L).projectId(31L).build());
        when(taskStreamService.createTask(eq(99L), eq(31L), eq("storyboard_episode_compose"), any(String.class), eq("storyboard_episode"), eq(11L), any(String.class)))
            .thenReturn("task-1");
        when(storyboardService.listScenesByEpisode(11L)).thenReturn(List.of(
                com.stonewu.fusion.entity.storyboard.StoryboardScene.builder().id(101L).sortOrder(0).build()
        ));
        when(storyboardService.listItemsByScene(101L)).thenReturn(List.of(
                com.stonewu.fusion.entity.storyboard.StoryboardItem.builder().id(201L).sortOrder(0).videoUrl("/media/videos/demo.mp4").build()
        ));
        when(episodeMapper.update(eq(null), any(UpdateWrapper.class))).thenReturn(1);

        String taskId = videoComposeService.submitCompose(11L, 99L);

        assertThat(taskId).isEqualTo("task-1");
        verify(episodeMapper).update(eq(null), any(UpdateWrapper.class));
        verify(videoComposeExecutor).execute(any(Runnable.class));
    }

    @Test
    void submitComposeMarksFailedWhenExecutorRejectsTask() {
        when(episodeMapper.selectById(11L)).thenReturn(StoryboardEpisode.builder().id(11L).storyboardId(21L).build());
        when(storyboardService.getById(21L)).thenReturn(Storyboard.builder().id(21L).projectId(31L).build());
        when(taskStreamService.createTask(eq(99L), eq(31L), eq("storyboard_episode_compose"), any(String.class), eq("storyboard_episode"), eq(11L), any(String.class)))
            .thenReturn("task-1");
        when(storyboardService.listScenesByEpisode(11L)).thenReturn(List.of(
                com.stonewu.fusion.entity.storyboard.StoryboardScene.builder().id(101L).sortOrder(0).build()
        ));
        when(storyboardService.listItemsByScene(101L)).thenReturn(List.of(
                com.stonewu.fusion.entity.storyboard.StoryboardItem.builder().id(201L).sortOrder(0).videoUrl("/media/videos/demo.mp4").build()
        ));
        when(episodeMapper.update(eq(null), any(UpdateWrapper.class))).thenReturn(1);
        org.mockito.Mockito.doThrow(new RejectedExecutionException("busy"))
                .when(videoComposeExecutor)
                .execute(any(Runnable.class));

        String taskId = videoComposeService.submitCompose(11L, 99L);

        assertThat(taskId).isEqualTo("task-1");
        ArgumentCaptor<StoryboardEpisode> captor = ArgumentCaptor.forClass(StoryboardEpisode.class);
        verify(episodeMapper).updateById(captor.capture());
        assertThat(captor.getValue().getId()).isEqualTo(11L);
        assertThat(captor.getValue().getComposeStatus()).isEqualTo(VideoComposeService.STATUS_FAILED);
        assertThat(captor.getValue().getComposeErrorMsg()).contains("合成队列繁忙");
        assertThat(captor.getValue().getComposedVideoUrl()).isNull();
        assertThat(captor.getValue().getComposedAt()).isNull();
        verify(taskStreamService).fail("task-1", "合成队列繁忙，请稍后重试");
    }

    @Test
    void validateRemoteUriRejectsLoopbackAddress() {
        assertThatThrownBy(() -> invokePrivate("validateRemoteUri", java.net.URI.create("http://127.0.0.1/test.mp4")))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("拒绝访问内网或本地地址");
    }

    @Test
    void resolveManagedMediaPathRejectsPathTraversal() {
        assertThatThrownBy(() -> invokePrivate("resolveManagedMediaPath", "/media/../../windows/system32/config/sam"))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("非法媒体相对路径");
    }

    @Test
    void resolveManagedMediaPathPrefersDefaultLocalStorageBasePath() throws Throwable {
        when(storageConfigService.getDefaultConfig()).thenReturn(StorageConfig.builder()
                .type("local")
                .basePath("D:/configured-media")
                .build());

        Object resolved = invokePrivate("resolveManagedMediaPath", "/media/videos/test.mp4");

        assertThat(resolved)
            .isEqualTo(java.nio.file.Paths.get("D:/configured-media/videos/test.mp4").toAbsolutePath().normalize());
    }

    @Test
    void resolveErrorMessageReturnsReadableHintWhenFfmpegExecutableMissing() throws Throwable {
        ReflectionTestUtils.setField(videoComposeService, "ffmpegPath", "C:/ffmpeg/bin/ffmpeg.exe");

        Method method = VideoComposeService.class.getDeclaredMethod("resolveErrorMessage", Throwable.class);
        method.setAccessible(true);
        String message = (String) method.invoke(
            videoComposeService,
            new IOException("Cannot run program \"ffmpeg\": CreateProcess error=2, 系统找不到指定的文件。")
        );

        assertThat(message)
                .contains("未找到 ffmpeg 可执行文件")
                .contains("video.compose.ffmpeg-path")
                .contains("C:/ffmpeg/bin/ffmpeg.exe");
    }

    private Object invokePrivate(String methodName, Object argument) throws Throwable {
        Method method = VideoComposeService.class.getDeclaredMethod(methodName, argument.getClass());
        method.setAccessible(true);
        try {
            return method.invoke(videoComposeService, argument);
        } catch (InvocationTargetException e) {
            throw e.getCause();
        }
    }
}