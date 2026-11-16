package com.stonewu.fusion.service.script;

import com.stonewu.fusion.service.resource.ScriptFields;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.PositiveOrZero;
import jakarta.validation.constraints.Size;
import org.springframework.stereotype.Component;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** 按标题和段落划分原文，偏移始终对应未经改写的 UTF-16 文本。 */
@Component
public class ScriptSourceSplitter {
    public static final int MAX_SOURCE_LENGTH = 20_000_000;
    public static final int MAX_EPISODE_LENGTH = 100_000;
    public static final int MAX_EPISODES = 20_000;
    private static final String NUMBER = "[0-9零〇一二三四五六七八九十百千万廿卅卌两兩壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+";
    private static final Pattern CHINESE = Pattern.compile("^(?:第)?(" + NUMBER + ")([集话話章回节節幕卷部篇])(?:[：:、.．\\-—]|$|(?=[^0-9]))(.*)$");
    private static final Pattern ENGLISH = Pattern.compile("^(chapter|chap\\.?|episode|ep\\.?|part|book|volume)(?:\\s+|(?=[0-9]))([0-9]+|[ivxlcdm]+)\\b[\\s.:：\\-]*(.*)$", Pattern.CASE_INSENSITIVE);
    private static final Pattern NUMBERED = Pattern.compile("^" + NUMBER + "[、.．:：]\\s*\\S.{0,70}$");
    private static final Pattern SPECIAL = Pattern.compile("^(序章|序言|前言|楔子|引子|序幕|尾声|尾聲|终章|終章|后记|後記|番外)(?:[\\s：:、—\\-].{0,70})?$");
    private static final Pattern TOC_ENTRY = Pattern.compile(".*(?:\\.{2,}|…{2,}|·{3,})\\s*\\d+\\s*$");

    public enum Rule { AUTO, CHAPTER, EPISODE, LENGTH }

    public record Segment(@PositiveOrZero int start, @PositiveOrZero int end,
                          @NotBlank @Size(max = 256) String title) {
    }

    public record Preview(List<Segment> segments, int totalChars, int headingCount, int splitCount) {
    }

    private record Heading(int start, int end, String title, int rank, int prefixStart) {
    }

    public Preview split(String source, Rule rule, int targetLength) {
        validateSource(source);
        if (rule == null || targetLength < 1_000 || targetLength > MAX_EPISODE_LENGTH) {
            throw ScriptFields.invalid("请选择分集方式，每集字数应在 1000 至 100000 之间");
        }
        List<Heading> headings = rule == Rule.LENGTH ? List.of() : headings(source);
        int rank = switch (rule) {
            case EPISODE -> 1;
            case CHAPTER -> 2;
            case AUTO -> headings.stream().filter(h -> h.rank() > 0).mapToInt(Heading::rank).min().orElse(0);
            case LENGTH -> 0;
        };
        List<Heading> selected = withoutTableOfContents(source,
                headings.stream().filter(h -> h.rank() == rank || h.rank() == 0)
                        .map(h -> rank < 4 && h.prefixStart() >= 0
                                ? new Heading(h.prefixStart(), h.end(), h.title(), h.rank(), -1) : h).toList());
        if (rule == Rule.CHAPTER || rule == Rule.EPISODE) {
            if (selected.stream().noneMatch(h -> h.rank() == rank)) {
                throw ScriptFields.invalid("未识别到所选标题，请选择自动识别或按字数分集");
            }
        }
        List<Segment> result = new ArrayList<>();
        int start = 0;
        String title = "第 1 集";
        if (!selected.isEmpty()) {
            Heading first = selected.getFirst();
            if (!source.substring(0, first.start()).isBlank()) {
                append(result, source, 0, first.start(), "前言", targetLength);
                start = first.start();
            }
            title = first.title();
            for (int i = 1; i < selected.size(); i++) {
                Heading next = selected.get(i);
                append(result, source, start, next.start(), title, targetLength);
                start = next.start();
                title = next.title();
            }
        }
        append(result, source, start, source.length(), title, targetLength);
        validateSegments(source, result);
        return new Preview(List.copyOf(result), source.length(), selected.size(), Math.max(0, result.size() - Math.max(1, selected.size())));
    }

    private List<Heading> headings(String source) {
        List<Heading> result = new ArrayList<>();
        String volume = "";
        int volumeStart = -1;
        int cursor = 0;
        while (cursor < source.length()) {
            int end = cursor;
            while (end < source.length() && source.charAt(end) != '\n' && source.charAt(end) != '\r') {
                end++;
            }
            String line = source.substring(cursor, end);
            // 仅检查短标题行，避免在百万字单行正文上执行模式匹配。
            if (line.length() <= 160) {
                String normalized = Normalizer.normalize(line, Normalizer.Form.NFKC).replace("\uFEFF", "").strip();
                normalized = normalized.replaceFirst("^#{1,6}\\s*", "").strip();
                String undecorated = normalized.replaceAll("[【】\\[\\]（）()]", "").strip();
                String compact = undecorated.replaceAll("[\\s\\p{Zs}]+", "");
                Matcher chinese = CHINESE.matcher(compact);
                Matcher english = ENGLISH.matcher(undecorated);
                int rank = -1;
                if (!TOC_ENTRY.matcher(normalized).matches() && chinese.matches()) {
                    rank = switch (chinese.group(2)) {
                        case "集", "话", "話" -> 1;
                        case "章", "回" -> 2;
                        case "节", "節", "幕" -> 3;
                        default -> 4;
                    };
                } else if (english.matches()) {
                    rank = switch (english.group(1).toLowerCase(Locale.ROOT).replace(".", "")) {
                        case "episode", "ep" -> 1;
                        case "chapter", "chap" -> 2;
                        default -> 4;
                    };
                } else if (SPECIAL.matcher(undecorated).matches()) {
                    rank = 0;
                } else if (NUMBERED.matcher(undecorated).matches()) {
                    rank = 5;
                }
                if (rank >= 0) {
                    if (rank == 4) {
                        volume = normalized;
                        volumeStart = cursor;
                    }
                    String title = rank >= 0 && rank < 4 && !volume.isEmpty() ? volume + " · " + normalized : normalized;
                    int prefixStart = rank >= 0 && rank < 4 ? volumeStart : -1;
                    if (rank >= 0 && rank < 4) {
                        volumeStart = -1;
                    }
                    result.add(new Heading(cursor, end, title, rank, prefixStart));
                }
            }
            cursor = end;
            if (cursor < source.length() && source.charAt(cursor) == '\r') {
                cursor++;
            }
            if (cursor < source.length() && source.charAt(cursor) == '\n') {
                cursor++;
            }
        }
        return result.stream().sorted(Comparator.comparingInt(Heading::start)).toList();
    }

    private List<Heading> withoutTableOfContents(String source, List<Heading> headings) {
        var laterTitles = new HashSet<String>();
        List<Heading> result = new ArrayList<>();
        for (int i = headings.size() - 1; i >= 0; i--) {
            Heading heading = headings.get(i);
            String title = heading.title().replaceAll("[\\s\\p{Zs}]+", "").toLowerCase(Locale.ROOT);
            int next = i + 1 < headings.size() ? headings.get(i + 1).start() : source.length();
            boolean repeated = !laterTitles.add(title);
            // 无正文的连续目录项不生成空分集；目录文字仍保留在前言中。
            if (!repeated || !source.substring(heading.end(), next).isBlank()) {
                result.add(heading);
            }
        }
        Collections.reverse(result);
        return result;
    }

    private void append(List<Segment> result, String source, int start, int end, String title, int target) {
        int part = 1;
        boolean divided = end - start > target;
        while (start < end) {
            int cut = Math.min(start + target, end);
            if (cut < end) {
                int paragraph = source.lastIndexOf('\n', cut - 1);
                if (paragraph >= start + target / 2) {
                    cut = paragraph + 1;
                } else {
                    int sentence = cut - 1;
                    while (sentence > start + target / 2 && "。！？.!?\r".indexOf(source.charAt(sentence)) < 0) {
                        sentence--;
                    }
                    if (sentence > start + target / 2) {
                        cut = sentence + 1;
                    }
                }
                if (Character.isHighSurrogate(source.charAt(cut - 1)) || source.charAt(cut - 1) == '\r' && source.charAt(cut) == '\n') {
                    cut--;
                }
            }
            result.add(new Segment(start, cut, divided ? title + "（" + part + "）" : title));
            start = cut;
            part++;
        }
    }

    public void validateSource(String source) {
        if (source == null || source.isBlank() || source.length() > MAX_SOURCE_LENGTH) {
            throw ScriptFields.invalid("请提供 2000 万字以内的原文");
        }
    }

    public void validateSegments(String source, List<Segment> segments) {
        validateSource(source);
        if (segments == null || segments.isEmpty() || segments.size() > MAX_EPISODES) {
            throw ScriptFields.invalid("分集数量应为 1 至 20000 集");
        }
        int previous = 0;
        for (Segment segment : segments) {
            if (segment == null || segment.start() != previous || segment.end() <= segment.start() || segment.end() > source.length()) {
                throw ScriptFields.invalid("分集范围不连续，请重新预览");
            }
            if (segment.end() - segment.start() > MAX_EPISODE_LENGTH) {
                throw ScriptFields.invalid("单集不能超过 10 万字，请继续拆分");
            }
            if (segment.end() < source.length() && Character.isHighSurrogate(source.charAt(segment.end() - 1))) {
                throw ScriptFields.invalid("分界位置位于字符中间，请调整分界");
            }
            if (segment.title() == null || segment.title().isBlank() || segment.title().length() > 256) {
                throw ScriptFields.invalid("请填写 256 字以内的分集标题");
            }
            previous = segment.end();
        }
        if (previous != source.length()) {
            throw ScriptFields.invalid("分集未包含全部原文，请重新预览");
        }
    }
}
