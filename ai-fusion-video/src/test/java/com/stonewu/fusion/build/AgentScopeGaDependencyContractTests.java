package com.stonewu.fusion.build;

import org.junit.jupiter.api.Test;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeGaDependencyContractTests {

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
        String pom = Files.readString(Path.of("pom.xml"));
        assertThat(pom).containsOnlyOnce("<agentscope.version>2.0.0</agentscope.version>");
        assertThat(pom).doesNotContain("agentscope-spring-boot-starter",
                "agentscope-extensions-session-mysql", "2.0.0-RC",
                "<artifactId>agentscope</artifactId>", "<artifactId>agentscope-core</artifactId>",
                "<jackson-bom.version>",
                "<artifactId>json-schema-validator</artifactId>");

        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        NodeList dependencies = factory.newDocumentBuilder()
                .parse(Path.of("pom.xml").toFile())
                .getElementsByTagName("dependency");
        List<String> agentScopeDependencies = new java.util.ArrayList<>();
        for (int index = 0; index < dependencies.getLength(); index++) {
            Element dependency = (Element) dependencies.item(index);
            if ("io.agentscope".equals(directChildText(dependency, "groupId"))) {
                agentScopeDependencies.add(directChildText(dependency, "artifactId")
                        + ":" + directChildText(dependency, "version"));
            }
        }
        assertThat(agentScopeDependencies).containsExactlyInAnyOrderElementsOf(EXPECTED_AGENTSCOPE_DEPENDENCIES);
    }

    private String directChildText(Element parent, String tagName) {
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
                Files.walk(Path.of("src/main/java")),
                Files.walk(Path.of("src/test/java")))) {
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
