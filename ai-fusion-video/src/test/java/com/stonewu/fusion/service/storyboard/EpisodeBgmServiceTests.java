package com.stonewu.fusion.service.storyboard;

import com.stonewu.fusion.config.EpisodeBgmProperties;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.sonilo.SoniloAudioClient;
import com.stonewu.fusion.service.storage.MediaStorageService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EpisodeBgmServiceTests {

    @Mock
    private SoniloAudioClient soniloAudioClient;

    @Mock
    private MediaStorageService mediaStorageService;

    @Mock
    private ApiConfigService apiConfigService;

    private EpisodeBgmProperties properties;
    private EpisodeBgmService episodeBgmService;

    private final Path video = Paths.get("/tmp/output.mp4");
    private final Path music = Paths.get("/tmp/bgm_music.m4a");
    private final Path sfx = Paths.get("/tmp/bgm_sfx.m4a");
    private final Path output = Paths.get("/tmp/output_bgm.mp4");

    @BeforeEach
    void setUp() {
        properties = new EpisodeBgmProperties();
        properties.setMusicVolume(0.7);
        episodeBgmService = new EpisodeBgmService(properties, apiConfigService);
        ReflectionTestUtils.setField(episodeBgmService, "ffmpegPath", "ffmpeg");
    }

    @Test
    void resolveApiConfigReturnsNullWithoutEnabledSoniloConfig() {
        when(apiConfigService.getListByPlatform("sonilo")).thenReturn(List.of());

        assertThat(episodeBgmService.resolveApiConfig()).isNull();
    }

    @Test
    void resolveApiConfigReturnsTheEnabledSoniloConfig() {
        ApiConfig config = ApiConfig.builder().id(5L).name("Sonilo").platform("sonilo").apiKey("sk-1").build();
        when(apiConfigService.getListByPlatform("sonilo")).thenReturn(List.of(config));

        assertThat(episodeBgmService.resolveApiConfig()).isSameAs(config);
    }

    @Test
    void resolveApiConfigRejectsAmbiguousEnabledProviders() {
        ApiConfig newer = ApiConfig.builder().id(7L).name("Sonilo B").platform("sonilo").apiKey("sk-b").build();
        ApiConfig older = ApiConfig.builder().id(3L).name("Sonilo A").platform("sonilo").apiKey("sk-a").build();
        when(apiConfigService.getListByPlatform("sonilo")).thenReturn(List.of(newer, older));

        assertThatThrownBy(() -> episodeBgmService.resolveApiConfig()).hasMessageContaining("只启用一个");
    }

    @Test
    void musicSkipReasonEnforcesSixMinuteCap() {
        assertThat(episodeBgmService.musicSkipReason(EpisodeBgmService.MUSIC_MAX_SECONDS)).isNull();
        assertThat(episodeBgmService.musicSkipReason(EpisodeBgmService.MUSIC_MAX_SECONDS + 1))
                .contains("超过配乐上限");
        assertThat(episodeBgmService.musicSkipReason(0)).contains("无法确定");
    }

    @Test
    void sfxAllowedEnforcesThreeMinuteCap() {
        assertThat(episodeBgmService.sfxAllowed(EpisodeBgmService.SFX_MAX_SECONDS)).isTrue();
        assertThat(episodeBgmService.sfxAllowed(EpisodeBgmService.SFX_MAX_SECONDS + 1)).isFalse();
    }

    @Test
    void mixCommandMapsMusicDirectlyWhenNoOtherAudio() {
        List<String> command = episodeBgmService.buildMixCommand(video, music, null, false, 4, output);

        assertThat(command).containsSubsequence("-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac");
        assertThat(command.get(command.indexOf("-filter_complex") + 1)).contains("volume=0.70,apad");
        assertThat(command).containsSubsequence("-t", "4.000").doesNotContain("-shortest");
    }

    @Test
    void mixCommandMixesWithOriginalAudioAtConfiguredVolume() {
        List<String> command = episodeBgmService.buildMixCommand(video, music, null, true, 4, output);

        int filterIndex = command.indexOf("-filter_complex");
        assertThat(filterIndex).isPositive();
        String filter = command.get(filterIndex + 1);
        assertThat(filter)
                .contains("[1:a]volume=0.70[bgm]")
                .contains("[0:a][bgm]amix=inputs=2:duration=longest:normalize=0,apad[aout]");
        assertThat(command).containsSubsequence("-map", "0:v:0", "-map", "[aout]", "-c:v", "copy");
    }

    @Test
    void mixCommandMixesThreeTracksWhenSfxPresent() {
        List<String> command = episodeBgmService.buildMixCommand(video, music, sfx, true, 4, output);

        String filter = command.get(command.indexOf("-filter_complex") + 1);
        assertThat(filter).contains("amix=inputs=3");
        assertThat(command).containsSubsequence("-i", video.toString(), "-i", music.toString(), "-i", sfx.toString());
    }
}
