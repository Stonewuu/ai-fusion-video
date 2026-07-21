package com.stonewu.fusion.build;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeGaDependencyContractTests {

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
