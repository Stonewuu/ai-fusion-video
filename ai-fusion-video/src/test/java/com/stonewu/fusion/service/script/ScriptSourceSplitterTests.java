package com.stonewu.fusion.service.script;

import com.stonewu.fusion.service.script.ScriptSourceSplitter.Rule;
import com.stonewu.fusion.service.script.ScriptSourceSplitter.Segment;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.time.Duration;
import java.util.List;
import com.stonewu.fusion.controller.script.vo.ScriptSourcePreviewReqVO;
import com.stonewu.fusion.controller.script.vo.ScriptSourceReplaceReqVO;
import com.stonewu.fusion.service.resource.model.ScriptSourceCommand;
import jakarta.validation.Validation;
import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTimeout;

class ScriptSourceSplitterTests {
    private final ScriptSourceSplitter splitter = new ScriptSourceSplitter();

    @ParameterizedTest
    @ValueSource(strings = {"第一章 云起", "第 壹佰零二 章：云起", "第１２章 云起", "第 1 2 集 云起", "第兩百回 云起",
            "CHAPTER IV: Dawn", "Episode 002 — Dawn", "## 第三章 云起", "【第一章】", "一、云起",
            "EP01: Dawn", "chapter12 Dawn", "第廿三話 云起"})
    void recognizesHeadingFormsWithoutChangingSource(String heading) {
        String source = heading + "\r\n林舟回到村庄。\r\n" + heading + "\n他找到了旧钥匙。";
        var result = splitter.split(source, Rule.AUTO, 12000);
        assertThat(result.segments()).hasSize(2);
        assertThat(join(source, result.segments())).isEqualTo(source);
        assertThat(result.headingCount()).isEqualTo(2);
    }

    @Test
    void preservesPrefaceTrailingWhitespaceAndDuplicateHeadings() {
        String source = "作品说明\n\n第一章 归来\n故事甲。\n第一章 归来\n故事乙。\n\n ";
        var result = splitter.split(source, Rule.AUTO, 12000);
        assertThat(result.segments()).extracting(Segment::title).containsExactly("前言", "第一章 归来", "第一章 归来");
        assertThat(join(source, result.segments())).isEqualTo(source);
    }

    @Test
    void prefersChapterBoundariesOverVolumeAndSectionNumbers() {
        String source = "第一卷 春\n第一章 归来\n正文。\n第一节 起点\n正文。\n第二章 出发\n正文。\n第二卷 夏\n第一章 归来\n正文。";
        var result = splitter.split(source, Rule.AUTO, 12000);
        assertThat(result.segments()).extracting(Segment::title).contains("第一卷 春 · 第一章 归来", "第二卷 夏 · 第一章 归来");
        assertThat(join(source, result.segments())).isEqualTo(source);
        assertThat(result.segments()).extracting(Segment::title).doesNotContain("第一节 起点");
    }

    @Test
    void volumePrologueAndChapterBoundariesNeverOverlap() {
        String source = "第一卷 春\n序章\n往事。\n第一章 归来\n正文。\n第二卷 夏\n序章\n回忆。\n第一章 出发\n结尾。";
        var result = splitter.split(source, Rule.AUTO, 12000);
        assertThat(result.segments()).hasSize(4);
        assertThat(join(source, result.segments())).isEqualTo(source);
        assertThat(source.substring(result.segments().get(2).start())).startsWith("第二卷 夏");
        String volumesOnly = "第一卷 春\n序章\n往事。\n第二卷 夏\n尾声\n终。";
        assertThat(join(volumesOnly, splitter.split(volumesOnly, Rule.AUTO, 12000).segments())).isEqualTo(volumesOnly);
    }

    @Test
    void splitsMillionCharacterSingleLineWithinBudgetAndWithoutBreakingUnicode() {
        String source = "正文𠮷🌕".repeat(220_000);
        var result = assertTimeout(Duration.ofSeconds(5), () -> splitter.split(source, Rule.AUTO, 12000));
        assertThat(result.headingCount()).isZero();
        assertThat(result.segments()).allSatisfy(segment -> {
            assertThat(segment.end() - segment.start()).isLessThanOrEqualTo(12000);
            assertThat(Character.isLowSurrogate(source.charAt(segment.start()))).isFalse();
            assertThat(Character.isHighSurrogate(source.charAt(segment.end() - 1))).isFalse();
        });
        assertThat(join(source, result.segments())).isEqualTo(source);
    }

    @Test
    void longChapterCutsAtParagraphsWithoutDroppingLineEndings() {
        String source = "第一章\r\n" + ("一段内容。".repeat(120) + "\r\n").repeat(40);
        var result = splitter.split(source, Rule.AUTO, 1000);
        assertThat(result.segments()).hasSizeGreaterThan(1);
        assertThat(join(source, result.segments())).isEqualTo(source);
        assertThat(result.segments().getFirst().end()).isEqualTo(source.indexOf("\r\n", 5) + 2);
    }

    @Test
    void ignoresDottedTableOfContentsAndQuotedChapterMentions() {
        String source = "第一章 开始......1\n第二章 出发......5\n\n第一章 开始\n他说：‘第二章还没写。’\n第二章 出发\n正文。";
        var result = splitter.split(source, Rule.AUTO, 12000);
        assertThat(result.headingCount()).isEqualTo(2);
        assertThat(join(source, result.segments())).isEqualTo(source);
    }

    @Test
    void repeatedContentsListDoesNotCreateEmptyEpisodesOrLoseText() {
        String source = "目录\n第一章 归来\n第二章 出发\n\n第一章 归来\n剧情甲。\n第二章 出发\n剧情乙。";
        var result = splitter.split(source, Rule.AUTO, 12000);
        assertThat(result.segments()).extracting(Segment::title).containsExactly("前言", "第一章 归来", "第二章 出发");
        assertThat(join(source, result.segments())).isEqualTo(source);
    }

    @Test
    void reviewedRangesMustCoverEveryCharacterExactlyOnceAndRemainBounded() {
        assertThatThrownBy(() -> splitter.validateSegments("原文内容", List.of(new Segment(0, 2, "甲"), new Segment(3, 4, "乙"))))
                .hasMessageContaining("不连续");
        assertThatThrownBy(() -> splitter.validateSegments("原文内容", List.of(new Segment(0, 3, "甲"))))
                .hasMessageContaining("全部原文");
        assertThatThrownBy(() -> splitter.validateSegments("🌕文字", List.of(new Segment(0, 1, "甲"), new Segment(1, 4, "乙"))))
                .hasMessageContaining("字符中间");
        assertThatThrownBy(() -> splitter.validateSegments("文".repeat(100001), List.of(new Segment(0, 100001, "甲"))))
                .hasMessageContaining("10 万");
        assertThatThrownBy(() -> splitter.split("未标章节的正文", Rule.CHAPTER, 12000)).hasMessageContaining("未识别");
    }

    @Test
    void largeNovelAndLongChaptersPassPreviewAndCommitBoundariesWithoutTextLoss() {
        var source = new StringBuilder();
        for (int i = 1; i <= 100; i++) {
            source.append("第 ").append(i).append(" 章 归途\n").append("小说正文。".repeat(8400)).append('\n');
        }
        String original = source.toString();
        var preview = splitter.split(original, Rule.AUTO, 100000);
        assertThat(preview.segments()).hasSize(100);
        assertThat(join(original, preview.segments())).isEqualTo(original);
        try (var factory = Validation.buildDefaultValidatorFactory()) {
            var validator = factory.getValidator();
            assertThat(validator.validate(new ScriptSourcePreviewReqVO(original, Rule.AUTO, 100000))).isEmpty();
            assertThat(validator.validate(new ScriptSourceReplaceReqVO("large-source", 0, original,
                    ScriptSourceCommand.Mode.APPEND, preview.segments()))).isEmpty();
            assertThat(validator.validate(new ScriptSourceCommand("large-source", 1, 0, original,
                    ScriptSourceCommand.Mode.APPEND, preview.segments()))).isEmpty();
        }
    }

    private String join(String source, List<Segment> segments) {
        var joined = new StringBuilder();
        segments.forEach(segment -> joined.append(source, segment.start(), segment.end()));
        return joined.toString();
    }
}
