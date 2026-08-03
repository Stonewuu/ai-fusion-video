package com.stonewu.fusion.service.project;

import com.stonewu.fusion.service.script.ScriptService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class ProjectWorkspaceCacheTests {

    private static final long MISSING_PROJECT_ID = Long.MAX_VALUE;

    @Autowired
    private ScriptService scriptService;

    @Autowired
    private StoryboardService storyboardService;

    @Test
    void missingProjectContainersReturnNullWithoutWritingNullToRedis() {
        assertThat(scriptService.getByProjectId(MISSING_PROJECT_ID)).isNull();
        assertThat(storyboardService.getByProjectId(MISSING_PROJECT_ID)).isNull();
    }
}
