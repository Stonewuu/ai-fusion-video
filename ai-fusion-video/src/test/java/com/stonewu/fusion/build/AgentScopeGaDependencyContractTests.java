package com.stonewu.fusion.build;

import org.junit.jupiter.api.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.CodeSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeGaDependencyContractTests {

    private static final String VICTOOLS_GROUP_ID = "com.github.victools";
    private static final String VICTOOLS_VERSION = "4.38.0";
    private static final Path PROJECT_ROOT = Path.of(System.getProperty("basedir"))
            .toAbsolutePath()
            .normalize();
    private static final Path POM_PATH = PROJECT_ROOT.resolve("pom.xml");

    private static final List<String> EXPECTED_AGENTSCOPE_DEPENDENCIES = List.of(
            "agentscope-harness:${agentscope.version}",
            "agentscope-extensions-redis:${agentscope.version}",
            "agentscope-extensions-model-openai:${agentscope.version}",
            "agentscope-extensions-model-anthropic:${agentscope.version}",
            "agentscope-extensions-model-gemini:${agentscope.version}",
            "agentscope-extensions-model-dashscope:${agentscope.version}",
            "agentscope-extensions-model-ollama:${agentscope.version}");

    private static final Pattern FORBIDDEN_SOURCE = Pattern.compile(
            "io\\.agentscope\\.core\\.(model\\.(OpenAIChatModel|AnthropicChatModel|GeminiChatModel|DashScopeChatModel|OllamaChatModel)(\\.Builder)?"
                    + "|formatter\\.(gemini|anthropic)\\.[A-Za-z0-9_$]+|session\\.mysql\\.[A-Za-z0-9_$]+)"
                    + "|AnthropicAgentScopeProxySupport|ProxyAwareAnthropicChatModel"
                    + "|GeminiToolResponseAwareChatFormatter|VertexAgentScopeProxySupport|MysqlSession");

    @Test
    void pomDeclaresOnlyGa() throws Exception {
        String pom = Files.readString(POM_PATH);
        assertThat(pom).containsOnlyOnce("<agentscope.version>2.0.0</agentscope.version>");
        assertThat(pom).containsOnlyOnce("<victools.version>4.38.0</victools.version>");
        assertThat(pom).doesNotContain("agentscope-spring-boot-starter",
                "agentscope-extensions-session-mysql", "2.0.0-RC",
                "<artifactId>agentscope</artifactId>", "<artifactId>agentscope-core</artifactId>",
                "<jackson-bom.version>",
                "<artifactId>json-schema-validator</artifactId>");

        List<String> agentScopeDependencies = new ArrayList<>();
        for (Element dependency : directDependencies(directChild(pomProject(), "dependencies"))) {
            if ("io.agentscope".equals(directChildText(dependency, "groupId"))) {
                agentScopeDependencies.add(directChildText(dependency, "artifactId")
                        + ":" + directChildText(dependency, "version"));
            }
        }
        assertThat(agentScopeDependencies).containsExactlyInAnyOrderElementsOf(EXPECTED_AGENTSCOPE_DEPENDENCIES);
    }

    @Test
    void pomImportsOfficialVictoolsBomAndRuntimeComponentsShareItsVersion() throws Exception {
        Element dependencyManagement = directChild(pomProject(), "dependencyManagement");
        List<Element> victoolsBomImports = directDependencies(directChild(dependencyManagement, "dependencies"))
                .stream()
                .filter(dependency -> VICTOOLS_GROUP_ID.equals(directChildText(dependency, "groupId")))
                .filter(dependency -> "jsonschema-generator-bom".equals(directChildText(dependency, "artifactId")))
                .toList();

        assertThat(victoolsBomImports).singleElement().satisfies(bom -> {
            assertThat(directChildText(bom, "version")).isEqualTo("${victools.version}");
            assertThat(directChildText(bom, "type")).isEqualTo("pom");
            assertThat(directChildText(bom, "scope")).isEqualTo("import");
        });
        for (VictoolsComponent component : List.of(
                new VictoolsComponent(com.github.victools.jsonschema.generator.SchemaGenerator.class,
                        "jsonschema-generator"),
                new VictoolsComponent(com.github.victools.jsonschema.module.jackson.JacksonModule.class,
                        "jsonschema-module-jackson"),
                new VictoolsComponent(com.github.victools.jsonschema.module.swagger2.Swagger2Module.class,
                        "jsonschema-module-swagger-2"))) {
            Properties metadata = mavenMetadata(component);
            assertThat(metadata.getProperty("groupId")).as(component.type().getName())
                    .isEqualTo(VICTOOLS_GROUP_ID);
            assertThat(metadata.getProperty("artifactId")).as(component.type().getName())
                    .isEqualTo(component.artifactId());
            assertThat(metadata.getProperty("version")).as(component.type().getName())
                    .isEqualTo(VICTOOLS_VERSION);
        }
    }

    private static Element pomProject() throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        Document document = factory.newDocumentBuilder().parse(POM_PATH.toFile());
        Element project = document.getDocumentElement();
        if (!"project".equals(project.getTagName())) {
            throw new AssertionError("Expected POM root <project> but found <" + project.getTagName() + ">");
        }
        return project;
    }

    private static Element directChild(Element parent, String tagName) {
        return directChildren(parent, tagName).stream()
                .findFirst()
                .orElseThrow(() -> new AssertionError("Missing direct <" + tagName + "> under <"
                        + parent.getTagName() + ">"));
    }

    private static List<Element> directDependencies(Element dependencies) {
        return directChildren(dependencies, "dependency");
    }

    private static List<Element> directChildren(Element parent, String tagName) {
        List<Element> elements = new ArrayList<>();
        NodeList children = parent.getChildNodes();
        for (int index = 0; index < children.getLength(); index++) {
            Node child = children.item(index);
            if (child instanceof Element element && tagName.equals(element.getTagName())) {
                elements.add(element);
            }
        }
        return elements;
    }

    private static Properties mavenMetadata(VictoolsComponent component) {
        CodeSource codeSource = component.type().getProtectionDomain().getCodeSource();
        if (codeSource == null) {
            throw new AssertionError("No CodeSource for " + component.type().getName());
        }

        Path jarPath;
        try {
            jarPath = Path.of(codeSource.getLocation().toURI());
        } catch (URISyntaxException failure) {
            throw new AssertionError("CodeSource is not a file JAR for " + component.type().getName(), failure);
        }
        if (!Files.isRegularFile(jarPath)) {
            throw new AssertionError("CodeSource is not a regular JAR file for " + component.type().getName()
                    + ": " + jarPath);
        }

        String metadataPath = "META-INF/maven/" + VICTOOLS_GROUP_ID + "/" + component.artifactId()
                + "/pom.properties";
        try (JarFile jar = new JarFile(jarPath.toFile())) {
            JarEntry metadata = jar.getJarEntry(metadataPath);
            if (metadata == null) {
                throw new AssertionError("Missing " + metadataPath + " in " + jarPath
                        + "; shaded or metadata-free JARs are not accepted");
            }
            Properties properties = new Properties();
            try (var input = jar.getInputStream(metadata)) {
                properties.load(input);
            }
            return properties;
        } catch (IOException failure) {
            throw new AssertionError("CodeSource is not a readable JAR for " + component.type().getName()
                    + ": " + jarPath, failure);
        }
    }

    private record VictoolsComponent(Class<?> type, String artifactId) {
    }

    private static String directChildText(Element parent, String tagName) {
        NodeList children = parent.getChildNodes();
        for (int index = 0; index < children.getLength(); index++) {
            Node child = children.item(index);
            if (child instanceof Element element && tagName.equals(element.getTagName())) {
                return element.getTextContent().trim();
            }
        }
        return "";
    }

    @Test
    void allOfficialModelExtensionsAndRedisApisLoad() throws Exception {
        for (String type : List.of(
                "io.agentscope.harness.agent.HarnessAgent",
                "io.agentscope.extensions.model.openai.OpenAIChatModel",
                "io.agentscope.extensions.model.anthropic.AnthropicChatModel",
                "io.agentscope.extensions.model.gemini.GeminiChatModel",
                "io.agentscope.extensions.model.dashscope.DashScopeChatModel",
                "io.agentscope.extensions.model.ollama.OllamaChatModel",
                "io.agentscope.extensions.redis.state.RedisAgentStateStore",
                "io.lettuce.core.RedisClient",
                "redis.clients.jedis.Jedis",
                "org.redisson.api.RedissonClient")) {
            assertThat(Class.forName(type)).isNotNull();
        }
    }

    @Test
    void sourceTreeContainsNoObsoleteV1Symbol() throws Exception {
        List<String> offenders;
        try (Stream<Path> files = Stream.concat(
                Files.walk(PROJECT_ROOT.resolve("src/main/java")),
                Files.walk(PROJECT_ROOT.resolve("src/test/java")))) {
            offenders = files.filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString()
                            .equals("AgentScopeGaDependencyContractTests.java"))
                    .filter(path -> {
                        try {
                            return FORBIDDEN_SOURCE.matcher(Files.readString(path)).find();
                        } catch (IOException failure) {
                            throw new UncheckedIOException(failure);
                        }
                    })
                    .map(Path::toString)
                    .sorted()
                    .toList();
        }
        assertThat(offenders).isEmpty();
    }
}
