package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.common.BusinessException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipInputStream;

@Service
@RequiredArgsConstructor
public class AgentSkillImportService {

    private static final long MAX_PACKAGE_BYTES = 30L * 1024 * 1024;
    private static final int MAX_PACKAGE_FILES = 1000;
    private static final int MAX_PATH_LENGTH = 512;
    private static final int MAX_PATH_DEPTH = 32;
    private static final Pattern WINDOWS_DRIVE_PATH = Pattern.compile("^[A-Za-z]:.*");
    private static final Pattern FRONTMATTER_PATTERN = Pattern.compile(
            "\\A---[ \\t]*\\r?\\n(.*?)\\r?\\n---[ \\t]*(?:\\r?\\n|\\z)(.*)\\z",
            Pattern.DOTALL);
    private static final Set<String> STANDARD_FIELDS = Set.of(
            "name", "description", "license", "compatibility", "metadata", "allowed-tools");

    private final AgentUserSkillService userSkillService;

    public ImportPreview preview(
            long userId,
            List<MultipartFile> uploads,
            List<String> paths) {
        Inspection inspection = inspect(uploads, paths);
        List<ImportCandidate> candidates = inspection.skills().stream()
                .map(skill -> skill.toPreview(
                        skill.name() != null
                                && skill.errors().isEmpty()
                                && userSkillService.exists(userId, skill.name())))
                .toList();
        return new ImportPreview(
                inspection.sourceType().name(),
                inspection.totalFiles(),
                inspection.totalBytes(),
                candidates,
                List.copyOf(inspection.warnings()),
                List.copyOf(inspection.errors()));
    }

    @Transactional
    @CacheEvict(value = "agentUserSkillCatalog", key = "#userId")
    public ImportResult importSkills(
            long userId,
            List<MultipartFile> uploads,
            List<String> paths,
            List<ImportSelection> selections) {
        Inspection inspection = inspect(uploads, paths);
        if (!inspection.errors().isEmpty()) {
            throw new BusinessException("导入包无效：" + String.join("；", inspection.errors()));
        }
        if (selections == null || selections.isEmpty()) {
            throw new BusinessException("请至少选择一个 Skill");
        }

        Map<String, InspectedSkill> candidates = inspection.skills().stream()
                .collect(Collectors.toMap(
                        InspectedSkill::rootPath,
                        skill -> skill,
                        (left, right) -> left,
                        LinkedHashMap::new));
        Set<String> selectedRoots = new HashSet<>();
        List<ReadyImport> ready = new ArrayList<>();
        List<String> skippedNames = new ArrayList<>();
        int createCount = 0;

        for (ImportSelection selection : selections) {
            if (selection == null || selection.rootPath() == null || selection.action() == null) {
                throw new BusinessException("导入选择项不完整");
            }
            if (!selectedRoots.add(selection.rootPath())) {
                throw new BusinessException("同一个 Skill 不能重复选择");
            }
            InspectedSkill candidate = candidates.get(selection.rootPath());
            if (candidate == null) {
                throw new BusinessException("导入包中不存在目录：" + selection.rootPath());
            }
            if (selection.action() == ImportAction.SKIP) {
                skippedNames.add(candidate.name() == null ? candidate.rootPath() : candidate.name());
                continue;
            }
            if (!candidate.errors().isEmpty() || candidate.name() == null) {
                throw new BusinessException(
                        "Skill " + candidate.rootPath() + " 未通过校验："
                                + String.join("；", candidate.errors()));
            }

            boolean exists = userSkillService.exists(userId, candidate.name());
            if (selection.action() == ImportAction.CREATE && exists) {
                throw new BusinessException("同名 Skill 已存在：" + candidate.name());
            }
            if (selection.action() == ImportAction.REPLACE && !exists) {
                throw new BusinessException("要覆盖的 Skill 已不存在，请重新预检：" + candidate.name());
            }
            String displayName = userSkillService.requireDisplayName(selection.displayName());
            if (!exists) {
                createCount++;
            }
            ready.add(new ReadyImport(candidate, displayName, exists));
        }

        if (userSkillService.skillCount(userId) + createCount
                > AgentUserSkillService.MAX_SKILLS_PER_USER) {
            throw new BusinessException("导入后将超过每个用户最多 64 个 Skill 的限制");
        }

        List<String> createdNames = new ArrayList<>();
        List<String> replacedNames = new ArrayList<>();
        for (ReadyImport item : ready) {
            userSkillService.replaceImportedDirectory(
                    userId,
                    item.skill().name(),
                    item.displayName(),
                    item.skill().files());
            if (item.replacing()) {
                replacedNames.add(item.skill().name());
            } else {
                createdNames.add(item.skill().name());
            }
        }
        return new ImportResult(
                createdNames.size(),
                replacedNames.size(),
                skippedNames.size(),
                List.copyOf(createdNames),
                List.copyOf(replacedNames),
                List.copyOf(skippedNames));
    }

    private Inspection inspect(List<MultipartFile> uploads, List<String> paths) {
        PreparedPackage prepared = preparePackage(uploads, paths);
        List<String> warnings = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        if (prepared.ignoredFiles() > 0) {
            warnings.add("已忽略 " + prepared.ignoredFiles() + " 个系统打包文件");
        }

        List<String> roots = prepared.files().keySet().stream()
                .filter(path -> path.endsWith("/SKILL.md"))
                .map(path -> path.substring(0, path.length() - "/SKILL.md".length()))
                .sorted()
                .toList();
        if (roots.isEmpty()) {
            errors.add("未找到位于 Skill 目录根部的 SKILL.md");
        }
        validateRootLayout(roots, errors);

        List<InspectedSkill> skills = roots.stream()
                .map(root -> inspectSkill(root, prepared.files()))
                .collect(Collectors.toCollection(ArrayList::new));
        Map<String, Long> duplicateNames = skills.stream()
                .filter(skill -> skill.name() != null)
                .collect(Collectors.groupingBy(InspectedSkill::name, LinkedHashMap::new, Collectors.counting()));
        skills.forEach(skill -> {
            if (skill.name() != null && duplicateNames.getOrDefault(skill.name(), 0L) > 1) {
                skill.errors().add("导入包中存在重复的 Skill 名称：" + skill.name());
            }
        });

        int outsideFiles = 0;
        for (String path : prepared.files().keySet()) {
            boolean included = roots.stream().anyMatch(root -> path.startsWith(root + "/"));
            if (!included) {
                outsideFiles++;
            }
        }
        if (outsideFiles > 0) {
            warnings.add("已忽略 Skill 目录外的 " + outsideFiles + " 个文件");
        }
        return new Inspection(
                prepared.sourceType(),
                prepared.files().size(),
                prepared.files().values().stream().mapToLong(bytes -> bytes.length).sum(),
                skills,
                warnings,
                errors);
    }

    private void validateRootLayout(List<String> roots, List<String> errors) {
        if (roots.isEmpty()) {
            return;
        }
        List<List<String>> segments = roots.stream()
                .map(root -> Arrays.asList(root.split("/")))
                .toList();
        if (segments.stream().anyMatch(parts -> parts.size() > 2)) {
            errors.add("Skill 目录最多只能位于压缩包顶层或一个公共包装目录下");
            return;
        }
        Set<Integer> depths = segments.stream().map(List::size).collect(Collectors.toSet());
        if (depths.size() > 1) {
            errors.add("不能混合顶层 Skill 与包装目录中的 Skill");
            return;
        }
        if (depths.contains(2)) {
            Set<String> wrappers = segments.stream().map(parts -> parts.getFirst()).collect(Collectors.toSet());
            if (wrappers.size() > 1) {
                errors.add("多个 Skill 必须位于同一个包装目录下");
            }
        }
    }

    private InspectedSkill inspectSkill(String root, Map<String, byte[]> packageFiles) {
        String prefix = root + "/";
        Map<String, byte[]> files = packageFiles.entrySet().stream()
                .filter(entry -> entry.getKey().startsWith(prefix))
                .collect(Collectors.toMap(
                        entry -> entry.getKey().substring(prefix.length()),
                        Map.Entry::getValue,
                        (left, right) -> left,
                        LinkedHashMap::new));
        List<String> warnings = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        byte[] skillBytes = files.get("SKILL.md");
        ParsedSkillDocument parsed = parseSkillDocument(root, skillBytes, warnings, errors);
        if (parsed.markdownBytes() != null) {
            files.put("SKILL.md", parsed.markdownBytes());
        }
        if (files.containsKey(AgentUserSkillService.METADATA_FILENAME)) {
            errors.add(AgentUserSkillService.METADATA_FILENAME + " 是系统保留文件，不能导入");
        }

        boolean hasScripts = hasPath(files, path -> path.startsWith("scripts/"));
        boolean hasReferences = hasPath(files, path -> path.startsWith("references/"));
        boolean hasAssets = hasPath(files, path -> path.startsWith("assets/"));
        if (hasScripts) {
            warnings.add("包含 scripts/：文件会被保存，但第一期不会执行脚本");
        }
        return new InspectedSkill(
                root,
                parsed.name(),
                parsed.description(),
                files.size(),
                files.values().stream().mapToLong(bytes -> bytes.length).sum(),
                hasScripts,
                hasReferences,
                hasAssets,
                files,
                warnings,
                errors);
    }

    private ParsedSkillDocument parseSkillDocument(
            String root,
            byte[] bytes,
            List<String> warnings,
            List<String> errors) {
        if (bytes == null) {
            errors.add("缺少 SKILL.md");
            return new ParsedSkillDocument(null, null, null);
        }
        if (bytes.length > AgentUserSkillService.MAX_CONTENT_LENGTH) {
            errors.add("SKILL.md 不能超过 256 KB");
            return new ParsedSkillDocument(null, null, null);
        }

        String markdown;
        try {
            markdown = decodeUtf8(bytes);
        } catch (CharacterCodingException invalidUtf8) {
            errors.add("SKILL.md 必须使用 UTF-8 编码");
            return new ParsedSkillDocument(null, null, null);
        }
        if (markdown.startsWith("\uFEFF")) {
            markdown = markdown.substring(1);
            warnings.add("已移除 SKILL.md 的 UTF-8 BOM");
        }
        Matcher matcher = FRONTMATTER_PATTERN.matcher(markdown);
        if (!matcher.matches()) {
            errors.add("SKILL.md 必须以 YAML frontmatter 开头");
            return new ParsedSkillDocument(null, null, markdown.getBytes(StandardCharsets.UTF_8));
        }

        Map<?, ?> metadata;
        try {
            LoaderOptions options = new LoaderOptions();
            options.setAllowDuplicateKeys(false);
            options.setMaxAliasesForCollections(10);
            options.setNestingDepthLimit(10);
            options.setCodePointLimit(16_384);
            Object value = new Yaml(new SafeConstructor(options)).load(matcher.group(1));
            if (!(value instanceof Map<?, ?> map)) {
                errors.add("SKILL.md frontmatter 必须是键值映射");
                return new ParsedSkillDocument(null, null, markdown.getBytes(StandardCharsets.UTF_8));
            }
            metadata = map;
        } catch (RuntimeException invalidYaml) {
            errors.add("SKILL.md frontmatter YAML 无效：" + conciseMessage(invalidYaml));
            return new ParsedSkillDocument(null, null, markdown.getBytes(StandardCharsets.UTF_8));
        }

        Object rawName = metadata.get("name");
        String name = rawName instanceof String text ? text : null;
        if (name == null || name.isBlank()) {
            errors.add("frontmatter.name 不能为空");
        } else if (codePointLength(name) > 64
                || !AgentUserSkillService.STANDARD_NAME_PATTERN.matcher(name).matches()) {
            errors.add("frontmatter.name 只能包含小写字母、数字和非连续短横线，最长 64 位");
        }

        Object rawDescription = metadata.get("description");
        String description = rawDescription instanceof String text ? text.trim() : null;
        if (description == null || description.isBlank()) {
            errors.add("frontmatter.description 不能为空");
        } else if (codePointLength(description) > AgentUserSkillService.MAX_DESCRIPTION_LENGTH) {
            errors.add("frontmatter.description 不能超过 1024 个字符");
        }

        validateOptionalString(metadata, "license", null, errors);
        validateOptionalString(metadata, "compatibility", 500, errors);
        validateOptionalString(metadata, "allowed-tools", null, errors);
        validateMetadataMap(metadata.get("metadata"), errors);
        if (metadata.keySet().stream().anyMatch(key -> !(key instanceof String))) {
            errors.add("SKILL.md frontmatter 的字段名必须是字符串");
        }
        metadata.keySet().stream()
                .filter(String.class::isInstance)
                .map(String.class::cast)
                .filter(key -> !STANDARD_FIELDS.contains(key))
                .sorted()
                .forEach(key -> warnings.add("将原样保留非标准 frontmatter 字段：" + key));
        if (metadata.containsKey("allowed-tools")) {
            warnings.add("allowed-tools 将原样保存，但不会自动授予工具权限");
        }

        if (name != null) {
            String directoryName = root.substring(root.lastIndexOf('/') + 1);
            if (!directoryName.equals(name)) {
                String anthropicCompatible = directoryName.toLowerCase(Locale.ROOT).replace('_', '-');
                if (anthropicCompatible.equals(name)) {
                    warnings.add("目录名将按 frontmatter.name 规范化为 " + name);
                } else {
                    errors.add("Skill 目录名必须与 frontmatter.name 匹配");
                }
            }
        }
        if (matcher.group(2).isBlank()) {
            errors.add("当前 AgentScope 运行时要求 SKILL.md 正文不能为空");
        }
        if (markdown.lines().count() > 500) {
            warnings.add("SKILL.md 超过规范建议的 500 行，建议拆分到 references/");
        }
        return new ParsedSkillDocument(
                name,
                description,
                markdown.getBytes(StandardCharsets.UTF_8));
    }

    private void validateOptionalString(
            Map<?, ?> metadata,
            String field,
            Integer maxLength,
            List<String> errors) {
        if (!metadata.containsKey(field)) {
            return;
        }
        Object value = metadata.get(field);
        if (!(value instanceof String text) || text.isBlank()) {
            errors.add("frontmatter." + field + " 必须是非空字符串");
        } else if (maxLength != null && codePointLength(text) > maxLength) {
            errors.add("frontmatter." + field + " 不能超过 " + maxLength + " 个字符");
        }
    }

    private void validateMetadataMap(Object value, List<String> errors) {
        if (value == null) {
            return;
        }
        if (!(value instanceof Map<?, ?> map)) {
            errors.add("frontmatter.metadata 必须是字符串键值映射");
            return;
        }
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String) || !(entry.getValue() instanceof String)) {
                errors.add("frontmatter.metadata 的键和值都必须是字符串");
                return;
            }
        }
    }

    private PreparedPackage preparePackage(List<MultipartFile> uploads, List<String> paths) {
        if (uploads == null || uploads.isEmpty()) {
            throw new BusinessException("请选择 ZIP 或 Skill 目录");
        }
        if (paths == null || uploads.size() != paths.size()) {
            throw new BusinessException("上传文件与相对路径数量不一致");
        }
        long uploadedBytes = uploads.stream().mapToLong(MultipartFile::getSize).sum();
        if (uploadedBytes > MAX_PACKAGE_BYTES) {
            throw new BusinessException("Skill 导入包不能超过 30 MB");
        }

        String onlyPath = paths.size() == 1 ? paths.getFirst() : null;
        boolean zip = onlyPath != null && onlyPath.toLowerCase(Locale.ROOT).endsWith(".zip");
        if (zip) {
            return expandZip(uploads.getFirst());
        }
        return collectDirectoryFiles(uploads, paths);
    }

    private PreparedPackage expandZip(MultipartFile upload) {
        Map<String, byte[]> files = new LinkedHashMap<>();
        Map<String, String> caseInsensitivePaths = new HashMap<>();
        int ignored = 0;
        int entryCount = 0;
        long totalBytes = 0;
        try (ZipInputStream zip = new ZipInputStream(upload.getInputStream(), StandardCharsets.UTF_8)) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                entryCount++;
                if (entryCount > MAX_PACKAGE_FILES) {
                    throw new BusinessException("Skill 导入包最多包含 1000 个目录项");
                }
                if (entry.isDirectory()) {
                    continue;
                }
                String path = normalizePath(entry.getName());
                if (isPackagingJunk(path)) {
                    ignored++;
                    continue;
                }
                byte[] bytes = readLimited(zip, MAX_PACKAGE_BYTES - totalBytes);
                totalBytes += bytes.length;
                addFile(files, caseInsensitivePaths, path, bytes);
            }
        } catch (BusinessException failure) {
            throw failure;
        } catch (ZipException invalidZip) {
            throw new BusinessException("ZIP 文件无效或使用了不支持的加密方式");
        } catch (IOException failure) {
            throw new BusinessException("读取 ZIP 文件失败：" + conciseMessage(failure));
        }
        if (files.isEmpty()) {
            throw new BusinessException("ZIP 中没有可导入的文件");
        }
        return new PreparedPackage(SourceType.ZIP, files, ignored);
    }

    private PreparedPackage collectDirectoryFiles(
            List<MultipartFile> uploads,
            List<String> paths) {
        if (uploads.size() > MAX_PACKAGE_FILES) {
            throw new BusinessException("Skill 导入目录最多包含 1000 个文件");
        }
        Map<String, byte[]> files = new LinkedHashMap<>();
        Map<String, String> caseInsensitivePaths = new HashMap<>();
        int ignored = 0;
        long totalBytes = 0;
        for (int index = 0; index < uploads.size(); index++) {
            String path = normalizePath(paths.get(index));
            if (isPackagingJunk(path)) {
                ignored++;
                continue;
            }
            try (InputStream input = uploads.get(index).getInputStream()) {
                byte[] bytes = readLimited(input, MAX_PACKAGE_BYTES - totalBytes);
                totalBytes += bytes.length;
                addFile(files, caseInsensitivePaths, path, bytes);
            } catch (BusinessException failure) {
                throw failure;
            } catch (IOException failure) {
                throw new BusinessException("读取上传文件失败：" + conciseMessage(failure));
            }
        }
        if (files.isEmpty()) {
            throw new BusinessException("所选目录中没有可导入的文件");
        }
        return new PreparedPackage(SourceType.DIRECTORY, files, ignored);
    }

    private void addFile(
            Map<String, byte[]> files,
            Map<String, String> caseInsensitivePaths,
            String path,
            byte[] bytes) {
        if (files.putIfAbsent(path, bytes) != null) {
            throw new BusinessException("导入包中存在重复路径：" + path);
        }
        String lowerPath = path.toLowerCase(Locale.ROOT);
        String previous = caseInsensitivePaths.putIfAbsent(lowerPath, path);
        if (previous != null && !previous.equals(path)) {
            throw new BusinessException("导入包中存在大小写冲突路径：" + previous + " / " + path);
        }
    }

    private byte[] readLimited(InputStream input, long remainingBytes) throws IOException {
        if (remainingBytes < 0) {
            throw new BusinessException("Skill 导入包解压后不能超过 30 MB");
        }
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        long count = 0;
        int read;
        while ((read = input.read(buffer)) != -1) {
            count += read;
            if (count > remainingBytes) {
                throw new BusinessException("Skill 导入包解压后不能超过 30 MB");
            }
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private String normalizePath(String rawPath) {
        if (rawPath == null
                || rawPath.isBlank()
                || rawPath.codePoints().anyMatch(Character::isISOControl)) {
            throw new BusinessException("导入包包含空路径或非法路径");
        }
        String path = rawPath.replace('\\', '/');
        if (path.startsWith("/") || WINDOWS_DRIVE_PATH.matcher(path).matches()) {
            throw new BusinessException("导入包不能包含绝对路径：" + rawPath);
        }
        while (path.startsWith("./")) {
            path = path.substring(2);
        }
        path = Normalizer.normalize(path, Normalizer.Form.NFC);
        String[] segments = path.split("/", -1);
        if (path.length() > MAX_PATH_LENGTH || segments.length > MAX_PATH_DEPTH) {
            throw new BusinessException("导入包路径过长或目录层级过深：" + rawPath);
        }
        for (String segment : segments) {
            if (segment.isBlank() || ".".equals(segment) || "..".equals(segment)) {
                throw new BusinessException("导入包包含路径穿越或空目录段：" + rawPath);
            }
        }
        return String.join("/", segments);
    }

    private boolean isPackagingJunk(String path) {
        return Arrays.stream(path.split("/"))
                .anyMatch(segment -> "__MACOSX".equals(segment) || ".DS_Store".equals(segment));
    }

    private boolean hasPath(Map<String, byte[]> files, Predicate<String> predicate) {
        return files.keySet().stream().anyMatch(predicate);
    }

    private String decodeUtf8(byte[] bytes) throws CharacterCodingException {
        return StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString();
    }

    private static int codePointLength(String value) {
        return value.codePointCount(0, value.length());
    }

    private String conciseMessage(Throwable failure) {
        String message = failure.getMessage();
        if (message == null || message.isBlank()) {
            return failure.getClass().getSimpleName();
        }
        return message.lines().findFirst().orElse(message);
    }

    public enum ImportAction {
        CREATE,
        REPLACE,
        SKIP
    }

    public record ImportSelection(
            String rootPath,
            String displayName,
            ImportAction action) {
    }

    public record ImportPreview(
            String sourceType,
            int totalFiles,
            long totalBytes,
            List<ImportCandidate> skills,
            List<String> warnings,
            List<String> errors) {
    }

    public record ImportCandidate(
            String rootPath,
            String name,
            String description,
            String suggestedDisplayName,
            int fileCount,
            long totalBytes,
            boolean hasScripts,
            boolean hasReferences,
            boolean hasAssets,
            boolean exists,
            boolean valid,
            ImportAction recommendedAction,
            List<String> warnings,
            List<String> errors) {
    }

    public record ImportResult(
            int createdCount,
            int replacedCount,
            int skippedCount,
            List<String> createdNames,
            List<String> replacedNames,
            List<String> skippedNames) {
    }

    private enum SourceType {
        ZIP,
        DIRECTORY
    }

    private record PreparedPackage(
            SourceType sourceType,
            Map<String, byte[]> files,
            int ignoredFiles) {
    }

    private record Inspection(
            SourceType sourceType,
            int totalFiles,
            long totalBytes,
            List<InspectedSkill> skills,
            List<String> warnings,
            List<String> errors) {
    }

    private record ParsedSkillDocument(
            String name,
            String description,
            byte[] markdownBytes) {
    }

    private record ReadyImport(
            InspectedSkill skill,
            String displayName,
            boolean replacing) {
    }

    private record InspectedSkill(
            String rootPath,
            String name,
            String description,
            int fileCount,
            long totalBytes,
            boolean hasScripts,
            boolean hasReferences,
            boolean hasAssets,
            Map<String, byte[]> files,
            List<String> warnings,
            List<String> errors) {

        private ImportCandidate toPreview(boolean exists) {
            return new ImportCandidate(
                    rootPath,
                    name,
                    description,
                    name == null ? rootPath.substring(rootPath.lastIndexOf('/') + 1) : name,
                    fileCount,
                    totalBytes,
                    hasScripts,
                    hasReferences,
                    hasAssets,
                    exists,
                    errors.isEmpty(),
                    exists ? ImportAction.SKIP : ImportAction.CREATE,
                    List.copyOf(warnings),
                    List.copyOf(errors));
        }
    }
}
