package com.stonewu.fusion.service.storyboard;

import com.stonewu.fusion.config.SoniloBgmProperties;
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

@ExtendWith(MockitoExtension.class)
class EpisodeBgmServiceTests {

    @Mock
    private SoniloAudioClient soniloAudioClient;

    @Mock
    private MediaStorageService mediaStorageService;

    private SoniloBgmProperties properties;
    private EpisodeBgmService episodeBgmService;

    private final Path video = Paths.get("/tmp/output.mp4");
    private final Path music = Paths.get("/tmp/bgm_music.m4a");
    private final Path sfx = Paths.get("/tmp/bgm_sfx.m4a");
    private final Path output = Paths.get("/tmp/output_bgm.mp4");

    @BeforeEach
    void setUp() {
        properties = new SoniloBgmProperties();
        properties.setMusicVolume(0.7);
        episodeBgmService = new EpisodeBgmService(properties, soniloAudioClient, mediaStorageService);
        ReflectionTestUtils.setField(episodeBgmService, "ffmpegPath", "ffmpeg");
        ReflectionTestUtils.setField(episodeBgmService, "ffprobePath", "ffprobe");
    }

    @Test
    void disabledByDefault() {
        assertThat(episodeBgmService.isEnabled()).isFalse();
    }

    @Test
    void musicSkipReasonEnforcesSixMinuteCap() {
        assertThat(episodeBgmService.musicSkipReason(EpisodeBgmService.MUSIC_MAX_SECONDS)).isNull();
        assertThat(episodeBgmService.musicSkipReason(EpisodeBgmService.MUSIC_MAX_SECONDS + 1))
                .contains("超过配乐上限");
        // 本地探测不到时长（0）时不拦截，交给后端强校验
        assertThat(episodeBgmService.musicSkipReason(0)).isNull();
    }

    @Test
    void sfxAllowedEnforcesThreeMinuteCap() {
        assertThat(episodeBgmService.sfxAllowed(EpisodeBgmService.SFX_MAX_SECONDS)).isTrue();
        assertThat(episodeBgmService.sfxAllowed(EpisodeBgmService.SFX_MAX_SECONDS + 1)).isFalse();
    }

    @Test
    void mixCommandMapsMusicDirectlyWhenNoOtherAudio() {
        List<String> command = episodeBgmService.buildMixCommand(video, music, null, false, output);

        assertThat(command).containsSubsequence("-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac");
        assertThat(command).doesNotContain("-filter_complex");
        assertThat(command).contains("-shortest");
    }

    @Test
    void mixCommandMixesWithOriginalAudioAtConfiguredVolume() {
        List<String> command = episodeBgmService.buildMixCommand(video, music, null, true, output);

        int filterIndex = command.indexOf("-filter_complex");
        assertThat(filterIndex).isPositive();
        String filter = command.get(filterIndex + 1);
        assertThat(filter)
                .contains("[1:a]volume=0.70[bgm]")
                .contains("[0:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]");
        assertThat(command).containsSubsequence("-map", "0:v", "-map", "[aout]", "-c:v", "copy");
    }

    @Test
    void mixCommandMixesThreeTracksWhenSfxPresent() {
        List<String> command = episodeBgmService.buildMixCommand(video, music, sfx, true, output);

        String filter = command.get(command.indexOf("-filter_complex") + 1);
        assertThat(filter).contains("amix=inputs=3");
        assertThat(command).containsSubsequence("-i", video.toString(), "-i", music.toString(), "-i", sfx.toString());
    }
}
