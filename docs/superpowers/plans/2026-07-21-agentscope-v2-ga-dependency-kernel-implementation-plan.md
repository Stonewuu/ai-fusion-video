# AgentScope V2 GA Dependency and Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a green backend baseline and a production-shaped AgentScope Java `2.0.0` GA kernel with bounded schedulers, typed per-call context, fail-closed shared state, hard-cap leased Harness reuse, and verified no-tool `call`/`streamEvents`.

**Architecture:** Spring owns one environment-specific `AgentStateStore` and every invocation owns a fresh `RuntimeContext`. `AgentKernelResource` owns Harness/model resources, `HarnessLeaseCache` enforces the hard capacity, and `DefaultAgentScopeHarnessInvoker` composes preflight, lease, execution, and complete/error/cancel cleanup with `usingWhen`. Journal/SSE, post-GA Provider behavior expansion, tool enablement, media, and Pipeline cutover remain separate plans.

**Tech Stack:** Java 21.0.2, Maven Wrapper 3.9.12, Spring Boot 3.5.14, Reactor, AgentScope Java 2.0.0 GA, Spring Data Redis, JUnit 5, AssertJ, Mockito, Reactor Test.

## Global Constraints

- Command working directories are fixed: run every Maven and `rg src/...` command from `ai-fusion-video/`; run every `git add`/`git commit` command from the repository root.
- All AgentScope coordinates use one `agentscope.version=2.0.0` property; no V1, RC, starter, session-mysql, or explicit `agentscope-core`.
- Upgrade `com.volcengine:volcengine-java-sdk-ark-runtime` to `2.0.19` before removing Jackson `2.17.3` and `json-schema-validator:3.0.0`.
- Keep Jedis, Lettuce, and Redisson transitives; add no Redis exclusions in this plan.
- Production code must not use `.block()`, `.toIterable()`, `Thread.sleep()`, or ThreadLocal.
- Scheduler queues are exact: state 512, journal 2048, model 256, tool 256.
- Thread defaults are exact: state/journal `max(4,min(32,CPU*2))`; model/tool `max(8,min(64,CPU*4))`.
- Cache defaults are exact: hard `maximumSize=64`, `expireAfterAccess=30m`, capacity wait `5s`; active entries are never evicted.
- Addressing is exact: stable authenticated userId and `sessionId=afv:v2:{conversationId}:{agentDefinitionStableKey}`; runId is not a sessionId.
- local/test share one `InMemoryAgentStateStore` per ApplicationContext; production uses Redis prefix `afv:agentscope:v2:`.
- Harness eviction never closes the Spring-owned Store; `SpringStringRedisClientAdapter.close()` is a no-op.
- Use `HarnessAgent.streamEvents(messages, runtimeContext)` and `getDelegate().interrupt(runtimeContext)`.
- V1 `agentscope:1.0.12` and V2 `agentscope-core:2.0.0` contain overlapping `io.agentscope.core.*` FQNs and must never share a classpath. Task 2 therefore performs one atomic compile cut: dependency replacement, removal of `MysqlSession`, Provider package/builder migration, and deletion of GA-covered formatter/proxy shims are staged and verified together before the commit.

---

### Task 1: Restore the green backend baseline

**Files:**
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/GenerationModelCapabilityServiceTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/tool/GetGenerationModelCapabilitiesToolExecutorTests.java`

**Interfaces:**
- Consumes: `new AiModelMetadataResolver(ApiConfigService)`.
- Produces: 111 passing pre-migration tests.

- [ ] **Step 1: Reproduce red**

Run: `cd ai-fusion-video; .\\mvnw.cmd test`

Expected: FAIL with 111 tests, 7 failures, 10 errors caused by null resolver fixtures.

- [ ] **Step 2: Replace every null resolver**

```java
private static GenerationModelCapabilityService capabilityService(ModelPresetService presets) {
    return new GenerationModelCapabilityService(
            new AiModelMetadataResolver(mock(ApiConfigService.class)),
            presets);
}
```

Use this helper in both classes, including anonymous preset services.

- [ ] **Step 3: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=GenerationModelCapabilityServiceTests,GetGenerationModelCapabilitiesToolExecutorTests" test; .\\mvnw.cmd test`

Expected: PASS, 111 tests, 0 failures, 0 errors.

```powershell
git add ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/GenerationModelCapabilityServiceTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/tool/GetGenerationModelCapabilitiesToolExecutorTests.java
git commit -m "test: restore green backend baseline"
```

---

### Task 2: Pin GA and Ark dependencies

**Files:**
- Modify: `ai-fusion-video/pom.xml`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeShutdownConfig.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeModelFactory.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentToolAdapter.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapter.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/StreamingEventHook.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AbstractAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProvider.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProviderContext.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProviderContextFactory.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProviderRegistry.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProviderService.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AnthropicAgentScopeProxySupport.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AnthropicAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/DashScopeAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/GeminiAiProvider.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/GeminiToolResponseAwareChatFormatter.java`
- Modify return type only: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/GoogleFlowReverseApiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/OllamaAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/OpenAiCompatibleAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/OpenAiResponsesAgentScopeModel.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/ProxyAwareAnthropicChatModel.java`
- Delete: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/VertexAgentScopeProxySupport.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/VertexAiProvider.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/proxy/AiProxySupport.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapterTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/AiModelServiceTests.java`
- Verify against GA without changing behavior: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/AiProviderContextFactoryTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/AnthropicAiProviderTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/DashScopeAiProviderTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/GeminiAiProviderTests.java`
- Delete: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/GeminiToolResponseAwareChatFormatterTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/OllamaAiProviderTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/OpenAiCompatibleAiProviderTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/VertexAiProviderTests.java`
- Delete: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/VertexAgentScopeProxySupportTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/build/AgentScopeGaDependencyContractTests.java`

**Interfaces:**
- Consumes: `agentscope-harness`, `agentscope-extensions-redis`, and all five official model extensions at exactly `2.0.0`.
- Produces: `ChatModelBase AiProvider.createAgentScopeModel(AiProviderContext)` and `ChatModelBase AiProviderService.createAgentScopeModel(AiModel)`.
- Produces: one atomic V1 `1.0.12` to V2 GA compile cut. There is no intermediate dependency-only commit and no classpath containing both generations, including during tests.

- [ ] **Step 1: Save evidence and write the red contract**

```java
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
```

Run: `.\\mvnw.cmd dependency:tree; .\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeGaDependencyContractTests" test`

Expected: FAIL because POM is V1 and Ark is 1.0.19.

- [ ] **Step 2: Replace dependency blocks**

```xml
<agentscope.version>2.0.0</agentscope.version>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-harness</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-extensions-redis</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-extensions-model-openai</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-extensions-model-anthropic</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-extensions-model-gemini</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-extensions-model-dashscope</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>io.agentscope</groupId><artifactId>agentscope-extensions-model-ollama</artifactId><version>\${agentscope.version}</version></dependency>
<dependency><groupId>com.volcengine</groupId><artifactId>volcengine-java-sdk-ark-runtime</artifactId><version>2.0.19</version></dependency>
```

Delete V1 starter/session, Jackson BOM property, and explicit schema-validator. Keep `openai-java:4.32.0`.

- [ ] **Step 3: In the same atomic patch, migrate the complete Provider/model surface**

```java
OpenAIChatModel.builder().apiKey(apiKey).modelName(model).stream(true)
    .baseUrl(baseUrl).endpointPath(endpointPath).generateOptions(options)
    .httpTransport(transport).proxy(proxy).build();
AnthropicChatModel.builder().apiKey(apiKey).modelName(model).stream(true)
    .defaultOptions(options).baseUrl(baseUrl).proxy(proxy).build();
GeminiChatModel.builder().apiKey(apiKey).modelName(model).streamEnabled(true)
    .project(project).location(location).vertexAI(vertex).credentials(credentials)
    .defaultOptions(options).proxy(proxy).build();
DashScopeChatModel.builder().apiKey(apiKey).modelName(model).stream(true)
    .enableThinking(enableThinking).defaultOptions(options).baseUrl(baseUrl)
    .httpTransport(transport).proxy(proxy).build();
OllamaChatModel.builder().modelName(model).baseUrl(baseUrl)
    .defaultOptions(ollamaOptions).httpTransport(transport).proxy(proxy).build();
```

Use these exact imports: `io.agentscope.extensions.model.openai.OpenAIChatModel`, `io.agentscope.extensions.model.anthropic.AnthropicChatModel`, `io.agentscope.extensions.model.gemini.GeminiChatModel`, `io.agentscope.extensions.model.dashscope.DashScopeChatModel`, and `io.agentscope.extensions.model.ollama.OllamaChatModel`. `OllamaAiProvider` maps configured `temperature` and `topP` into `io.agentscope.extensions.model.ollama.OllamaOptions.builder()`; Ollama has no stream builder method.

Change the complete Provider abstraction and every implementation to return `ChatModelBase`. `OpenAiResponsesAgentScopeModel` changes from `implements Model` to `extends ChatModelBase` and implements the GA `doStream(List<Msg>, List<ToolSchema>, GenerateOptions)` contract. `GoogleFlowReverseApiProvider` keeps throwing its existing unsupported-operation `BusinessException` but adopts the `ChatModelBase` return type. Change the transitional `AgentScopeModelFactory` cache and `getOrCreate(AiModel)` return type to `ChatModelBase`; Task 8 removes that cache when kernel resources own the lifetime.

In `AgentScopeAssistantService`, remove the `MysqlSession` and `DataSource` fields, the session `@PostConstruct` initializer, and every `.session(mysqlSession)` call. Retain and compile-check its GA-preserved `ReActAgent`, `ExecutionConfig`, `Toolkit`, and hook calls; do not treat an overlapping FQN as evidence that V1 is present. Compile-check `AgentScopeShutdownConfig`, both tool adapters, `StreamingEventHook`, `AbstractAiProvider`, and `AiProxySupport` against GA in this same cut.

For Vertex, use the official Gemini extension directly:

```java
GeminiChatModel.builder()
        .modelName(context.getModelCode())
        .project(projectId)
        .location(location)
        .vertexAI(true)
        .credentials(credentials)
        .defaultOptions(buildGeminiGenerateOptions(context))
        .proxy(AiProxySupport.agentScopeProxyConfig(context.getApiConfig()))
        .streamEnabled(true)
        .build();
```

First move Gemini tool-result ordering, Anthropic authenticated proxy, and Vertex credentials/thinking/proxy assertions into the Provider tests. Then delete the four compatibility classes and their two dedicated tests. Add positive type/options/proxy tests for OpenAI, Anthropic, Gemini, DashScope, Ollama, and Vertex; do not use only `isNotInstanceOf` assertions.

- [ ] **Step 4: Run the exact negative scans before the atomic compile gate**

Run from `ai-fusion-video`:

```powershell
$oldJava = rg -n --glob '*.java' --glob '!AgentScopeGaDependencyContractTests.java' 'io\.agentscope\.core\.(model\.(OpenAIChatModel|AnthropicChatModel|GeminiChatModel|DashScopeChatModel|OllamaChatModel)(\.Builder)?|formatter\.(gemini|anthropic)\.[A-Za-z0-9_$]+|session\.mysql\.[A-Za-z0-9_$]+)|AnthropicAgentScopeProxySupport|ProxyAwareAnthropicChatModel|GeminiToolResponseAwareChatFormatter|VertexAgentScopeProxySupport|MysqlSession' src/main src/test
if ($LASTEXITCODE -eq 0) { $oldJava; throw 'obsolete AgentScope V1 Java symbol remains' }
if ($LASTEXITCODE -ne 1) { throw 'AgentScope Java source scan failed' }
$oldPom = rg -n 'agentscope-spring-boot-starter|agentscope-extensions-session-mysql|<artifactId>agentscope</artifactId>|<artifactId>agentscope-core</artifactId>|2\.0\.0-RC|<jackson-bom\.version>|<artifactId>json-schema-validator</artifactId>' pom.xml
if ($LASTEXITCODE -eq 0) { $oldPom; throw 'obsolete AgentScope/Jackson dependency remains' }
if ($LASTEXITCODE -ne 1) { throw 'POM scan failed' }
```

Expected: both `rg` invocations return exit code 1 with no matches; the wrapper script exits successfully. The scan deliberately does not ban all `io.agentscope.core.*` imports because GA retains overlapping FQNs such as `Model`, `Msg`, `ReActAgent`, `AgentTool`, and `ToolCallParam`.

- [ ] **Step 5: Verify and commit the indivisible compile cut**

Run: `.\\mvnw.cmd dependency:tree "-Dincludes=io.agentscope,com.volcengine,io.lettuce,redis.clients,org.redisson"; .\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeGaDependencyContractTests,AgentScopeToolAdapterTests,AiModelServiceTests,AiProviderContextFactoryTests,OpenAiCompatibleAiProviderTests,AnthropicAiProviderTests,GeminiAiProviderTests,DashScopeAiProviderTests,OllamaAiProviderTests,VertexAiProviderTests" test; .\\mvnw.cmd test`

Expected: all three commands PASS. The tree contains only AgentScope `2.0.0`, Ark is `2.0.19`, all five official extension classes and all three Redis APIs load, the exact old-symbol scans are empty, and the full suite is green. A dependency-only or focused-test-only result is not committable.

```powershell
git add ai-fusion-video/pom.xml ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeShutdownConfig.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeModelFactory.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentToolAdapter.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapter.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/StreamingEventHook.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/proxy/AiProxySupport.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/AiModelServiceTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapterTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider ai-fusion-video/src/test/java/com/stonewu/fusion/build/AgentScopeGaDependencyContractTests.java
git commit -m "build: pin AgentScope 2.0.0 GA dependencies"
```

---

### Task 3: Freeze GA API and Ark/Jackson contracts

**Files:**
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/build/AgentScopeGaApiContractTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/build/ArkJacksonCompatibilityTests.java`

**Interfaces:**
- Produces: compile failures on runtime/message/tool/model API drift.

- [ ] **Step 1: Write contracts**

```java
private static final class EchoModel extends ChatModelBase {
    @Override protected Flux<ChatResponse> doStream(List<Msg> m, List<ToolSchema> t, GenerateOptions o) {
        return Flux.just(ChatResponse.builder().content(List.of()).finishReason("stop").build());
    }
    @Override public String getModelName() { return "echo"; }
}

@Test void gaContract() throws Exception {
    RuntimeContext c = RuntimeContext.builder().userId("42").sessionId("afv:v2:c:a").put(String.class, "typed").build();
    assertThat(c.get(String.class)).isEqualTo("typed");
    assertThat(ToolCallParam.class.getMethod("getRuntimeContext")).isNotNull();
    assertThat(Arrays.stream(ToolBase.Builder.class.getDeclaredMethods()).noneMatch(m -> m.getName().equals("build"))).isTrue();
    assertThat(HarnessAgent.builder().name("test").model(new EchoModel()).stateStore(new InMemoryAgentStateStore())).isNotNull();
}
```

```java
@Test void arkSerializes() throws Exception {
    GenerateImagesRequest r = GenerateImagesRequest.builder()
            .model("seedream-test").prompt("test").size("1024x1024").watermark(false).build();
    assertThat(new ObjectMapper().writeValueAsString(r)).contains("seedream-test");
}
```

- [ ] **Step 2: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeGaApiContractTests,ArkJacksonCompatibilityTests" test`

Expected: PASS for Harness, RuntimeContext null behavior, targeted interrupt compilation, ChatModelBase, messages/media, ToolBase, and Ark serialization.

```powershell
git add ai-fusion-video/src/test/java/com/stonewu/fusion/build
git commit -m "test: freeze AgentScope GA contracts"
```

---

### Task 4: Add four bounded schedulers

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeProperties.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/runtime/AgentRuntimeSchedulers.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/runtime/AgentRuntimeSchedulersTests.java`

**Interfaces:**
- Produces: `state()`, `journal()`, `modelBlocking()`, `toolBlocking()`, idempotent `close()`.

- [ ] **Step 1: Write red tests for exact defaults and thread names**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentRuntimeSchedulersTests" test`

Expected: FAIL to compile.

- [ ] **Step 2: Implement bounded ownership**

```java
private OwnedScheduler fixedScheduler(String name, int threads, int queueCapacity) {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(
            threads, threads, 0L, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(queueCapacity),
            namedThreadFactory(name),
            new ThreadPoolExecutor.AbortPolicy());
    executor.prestartAllCoreThreads();
    return new OwnedScheduler(executor, Schedulers.fromExecutorService(executor, name));
}

state = fixedScheduler("agent-state", stateThreads, 512);
journal = fixedScheduler("agent-journal", journalThreads, 2048);
modelBlocking = fixedScheduler("agent-model-blocking", modelThreads, 256);
toolBlocking = fixedScheduler("agent-tool-blocking", toolThreads, 256);
```

`close()` uses one `AtomicBoolean.compareAndSet(false,true)`, disposes each `Scheduler`, and shuts down each owned executor. A rejected submission is translated to the scheduler-specific overload code and is never retried on the caller thread.

- [ ] **Step 3: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentRuntimeSchedulersTests" test`

Expected: PASS and all four thread prefixes observed.

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeProperties.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/runtime ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/runtime
git commit -m "feat: add bounded AgentScope schedulers"
```

---

### Task 5: Create typed per-call RuntimeContext

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/AuthenticatedUserContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/AgentConversationContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/AgentRunContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/ProjectContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/PipelineRequestContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/ToolExecutionContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/CancellationContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/AgentScopeRuntimeContextRequest.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/AgentScopeRuntimeContextFactory.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/context/AgentScopeRuntimeContextFactoryTests.java`

**Interfaces:**
- Produces: `RuntimeContext create(AgentScopeRuntimeContextRequest)`.

- [ ] **Step 1: Write red addressing/null tests**

```java
RuntimeContext c = factory.create(request);
assertThat(c.getUserId()).isEqualTo("42");
assertThat(c.getSessionId()).isEqualTo("afv:v2:conversation-7:assistant-v3");
assertThat(c.get(AgentRunContext.class).ownerEpoch()).isEqualTo(3);
assertThat(c.get(ProjectContext.class)).isNull();
```

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeRuntimeContextFactoryTests" test`

Expected: FAIL to compile.

- [ ] **Step 2: Implement immutable records and factory**

```java
public record AgentRunContext(
        String runId,
        String ownerInstanceId,
        long ownerEpoch,
        Instant deadline) {}

RuntimeContext.Builder b = RuntimeContext.builder()
        .userId(String.valueOf(r.authenticatedUser().userId()))
        .sessionId("afv:v2:" + r.conversation().conversationId() + ":" + r.conversation().agentDefinitionStableKey())
        .put(AuthenticatedUserContext.class, r.authenticatedUser())
        .put(AgentConversationContext.class, r.conversation())
        .put(AgentRunContext.class, r.run())
        .put(PipelineRequestContext.class, r.pipelineRequest())
        .put(CancellationContext.class, r.cancellation());
if (r.project() != null) b.put(ProjectContext.class, r.project());
if (r.toolExecution() != null) b.put(ToolExecutionContext.class, r.toolExecution());
return b.build();
```

- [ ] **Step 3: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeRuntimeContextFactoryTests" test`

Expected: PASS; no null is passed to string-keyed builder `put`.

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/context
git commit -m "feat: add typed AgentScope runtime context"
```

---

### Task 6: Implement Spring Redis adapter and fail-closed state

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/SpringStringRedisClientAdapter.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/StateStoreSlot.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/StateStoreFailure.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/StateStoreFailureGuard.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/InMemoryStateStoreFailureGuard.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/FailClosedAgentStateStore.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state/SpringStringRedisClientAdapterTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state/StateStoreFailureGuardTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state/FailClosedAgentStateStoreTests.java`

**Interfaces:**
- Produces: all `RedisClientAdapter` methods and a full `AgentStateStore` decorator.
- Produces: one exception type, `StateStoreFailure`, for both stored failure state and the exception thrown by `throwIfFailed`.

- [ ] **Step 1: Write red tests**

Test set/get/list/set/delete/SCAN, adapter close no-op, bulkhead rejection, and exact-slot failure recording. Freeze the failure semantics: the first failure for a slot wins, later failures cannot replace it, `throwIfFailed` throws the same stored exception instance with the original cause, throwing does not clear the marker, and only explicit `clear(slot)` removes it.

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=SpringStringRedisClientAdapterTests,StateStoreFailureGuardTests,FailClosedAgentStateStoreTests" test`

Expected: FAIL to compile.

- [ ] **Step 2: Implement guarded delegation**

```java
public final class StateStoreFailure extends RuntimeException {
    private final StateStoreSlot slot;
    private final String operation;

    public StateStoreFailure(StateStoreSlot slot, String operation, Throwable cause) {
        super("AgentStateStore " + operation + " failed for " + slot, cause);
        this.slot = slot;
        this.operation = operation;
    }

    public StateStoreSlot slot() { return slot; }
    public String operation() { return operation; }
}

public interface StateStoreFailureGuard {
    void clear(StateStoreSlot slot);
    StateStoreFailure record(StateStoreSlot slot, String operation, Throwable cause);
    Optional<StateStoreFailure> failure(StateStoreSlot slot);
    void throwIfFailed(StateStoreSlot slot) throws StateStoreFailure;
}

public final class InMemoryStateStoreFailureGuard implements StateStoreFailureGuard {
    private final ConcurrentMap<StateStoreSlot, StateStoreFailure> failures = new ConcurrentHashMap<>();

    @Override
    public StateStoreFailure record(StateStoreSlot slot, String operation, Throwable cause) {
        StateStoreFailure candidate = new StateStoreFailure(slot, operation, cause);
        StateStoreFailure stored = failures.putIfAbsent(slot, candidate);
        return stored != null ? stored : candidate;
    }

    @Override
    public void throwIfFailed(StateStoreSlot slot) throws StateStoreFailure {
        StateStoreFailure stored = failures.get(slot);
        if (stored != null) throw stored;
    }
}

private <T> T guarded(String userId, String sessionId, String operation, Supplier<T> action) {
    try { return action.get(); }
    catch (RuntimeException failure) {
        throw failures.record(new StateStoreSlot(userId, sessionId), operation, failure);
    }
}
```

`StateStoreSlot` is the immutable `record StateStoreSlot(String userId, String sessionId)`. Delegate save/get/getList/exists/full-session delete/listSessionIds/close. Every delegate failure is wrapped exactly once as `StateStoreFailure`; if the slot already has a failure, rethrow that first stored instance. Adapter operations use `Semaphore.tryAcquire()` and throw `StateStoreFailure(slot, operation, new BusinessException(503, "STATE_STORE_FAILED: bulkhead full"))` through the decorator when saturated; SCAN uses `RedisCallback` and `Cursor`; adapter `close()` is empty.

- [ ] **Step 3: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=SpringStringRedisClientAdapterTests,StateStoreFailureGuardTests,FailClosedAgentStateStoreTests" test`

Expected: PASS; the exact stored `StateStoreFailure` instance and cause remain visible across `failure` and repeated `throwIfFailed` calls until `clear`.

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state
git commit -m "feat: fail closed AgentScope state access"
```

---

### Task 7: Wire shared stores, preflight, and model guard

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeConfiguration.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/AgentScopeStateStoreFactory.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/AgentStatePreflight.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/StateStoreGuardedChatModel.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/AgentConversationService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/AiAssistantController.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state/AgentScopeStateStoreFactoryTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state/AgentStatePreflightTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state/StateStoreGuardedChatModelTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/AgentConversationServiceTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/controller/ai/AiAssistantControllerTests.java`

**Interfaces:**
- Produces: one shared Store; `Mono<Void> AgentStatePreflight.check(RuntimeContext context)`; `Mono<Void> AgentStatePreflight.deleteConversationSessions(String runtimeUserId, String conversationId)`; guarded exact `doStream`.
- Changes: `Mono<Void> AgentConversationService.deleteConversation(long id, long currentUserId)` and `Mono<CommonResult<Boolean>> AiAssistantController.deleteConversation(Long id)`.

- [ ] **Step 1: Write red profile/fail-closed tests**

Assert one InMemory bean per local/test context, distinct beans across contexts, Redis factory prefix, and a delegate state load failure rejected before model invocation.

Also assert conversation deletion reads only a row owned by `currentUserId`, enumerates and removes every session with prefix `afv:v2:{conversationId}:`, performs idempotent state cleanup before deleting the row, leaves the row intact when cleanup fails so the request can be retried, and returns the controller response only after both phases complete. A missing/already-deleted row completes without exposing whether another user owned it.

- [ ] **Step 2: Implement preflight and guard**

```java
public Mono<Void> check(RuntimeContext c) {
    return Mono.fromRunnable(() -> {
        StateStoreSlot slot = new StateStoreSlot(c.getUserId(), c.getSessionId());
        failures.clear(slot);
        store.exists(c.getUserId(), c.getSessionId());
        failures.throwIfFailed(slot);
    }).subscribeOn(schedulers.state()).then();
}

public Mono<Void> deleteConversationSessions(String runtimeUserId, String conversationId) {
    String prefix = "afv:v2:" + conversationId + ":";
    return Mono.fromCallable(() -> store.listSessionIds(runtimeUserId).stream()
            .filter(sessionId -> sessionId.startsWith(prefix)).toList())
        .subscribeOn(schedulers.state())
        .flatMapMany(Flux::fromIterable)
        .concatMap(sessionId -> Mono.fromRunnable(() -> deleteWholeSession(runtimeUserId, sessionId))
            .subscribeOn(schedulers.state()))
        .then();
}

public Mono<Void> deleteConversation(long id, long currentUserId) {
    return Mono.fromCallable(() -> conversationMapper.selectOne(
            new LambdaQueryWrapper<AgentConversation>()
                    .eq(AgentConversation::getId, id)
                    .eq(AgentConversation::getUserId, currentUserId)))
        .subscribeOn(schedulers.journal())
        .flatMap(conversation -> statePreflight.deleteConversationSessions(
                String.valueOf(currentUserId), conversation.getConversationId())
            .then(Mono.fromRunnable(() -> conversationMapper.delete(
                    new LambdaQueryWrapper<AgentConversation>()
                            .eq(AgentConversation::getId, id)
                            .eq(AgentConversation::getUserId, currentUserId)))
                .subscribeOn(schedulers.journal())));
}

@DeleteMapping("/conversations/{id}")
public Mono<CommonResult<Boolean>> deleteConversation(@PathVariable Long id) {
    long currentUserId = requireCurrentUserId();
    return conversationService.deleteConversation(id, currentUserId)
            .thenReturn(CommonResult.success(true));
}
```

```java
return Flux.deferContextual(view -> {
    RuntimeContext c = view.getOrDefault(AgentBase.RUNTIME_CONTEXT_KEY, null);
    if (c == null) return Flux.error(new BusinessException(500, "STATE_STORE_FAILED: RuntimeContext missing"));
    failures.throwIfFailed(new StateStoreSlot(c.getUserId(), c.getSessionId()));
    return delegate.stream(messages, tools, options);
});
```

- [ ] **Step 3: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeStateStoreFactoryTests,AgentStatePreflightTests,StateStoreGuardedChatModelTests,AgentConversationServiceTests,AiAssistantControllerTests" test`

Expected: PASS; Redis failure never falls back to memory.

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeConfiguration.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/AgentConversationService.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/AiAssistantController.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/AgentConversationServiceTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/controller/ai/AiAssistantControllerTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/state
git commit -m "feat: wire shared AgentScope state store"
```

---

### Task 8: Build explicit kernel resources and no-tool Harness

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelToolManifest.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelKey.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelSpec.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/OwnedChatModel.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelModelFactory.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelToolkitResources.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelToolRegistry.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelResource.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentScopeHarnessFactory.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeConfiguration.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeModelFactory.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/AiModelService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/ApiConfigService.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelResourceTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentScopeHarnessFactoryTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelModelFactoryContextTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/AiModelServiceTests.java`

**Interfaces:**
- Produces: `OwnedChatModel AgentKernelModelFactory.create(AgentKernelSpec spec)` and `AgentKernelResource AgentScopeHarnessFactory.create(AgentKernelSpec spec)`.
- Produces: `AgentKernelToolkitResources AgentKernelToolRegistry.register(AgentKernelSpec spec, Toolkit toolkit)` as the future tool/sub-agent extension point.
- Produces: `HarnessAgent AgentKernelResource.agent()` and idempotent `AgentKernelResource.close()`; the Spring-owned Store is borrowed and is never closed by the resource.

- [ ] **Step 1: Write red ownership tests**

Assert changing any of agent definition, model configuration fingerprint, prompt version, tool manifest fingerprint, or whitelist version changes `AgentKernelKey`. Assert record collections are immutable and a spec whose key whitelist version differs from `toolWhitelistVersion` is rejected. Assert the registry is invoked exactly once, a test tool registered by it reaches the Harness `Toolkit`, resources returned by the registry close exactly once, no built-in platform tools are enabled, owned model/Harness closeables close once, and the shared Store never closes.

In `AgentKernelModelFactoryContextTests`, start an ApplicationContext and assert `context.getBeansOfType(AgentKernelModelFactory.class)` has size one, its only bean name is `agentScopeModelFactory`, and its value is an `AgentScopeModelFactory`.

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentKernelResourceTests,AgentScopeHarnessFactoryTests,AgentKernelModelFactoryContextTests" test`

Expected: FAIL to compile because the kernel records, registry, and factory interface do not exist.

- [ ] **Step 2: Freeze the key, spec, manifest, and registry contracts**

```java
public record AgentKernelToolManifest(
        String toolName,
        String schemaSha256,
        boolean readOnly,
        boolean concurrencySafe) {}

public record AgentKernelKey(
        String agentDefinitionStableKey,
        String modelConfigFingerprint,
        String promptVersion,
        String toolManifestFingerprint,
        String toolWhitelistVersion) {}

public record AgentKernelSpec(
        AgentKernelKey key,
        AiModel model,
        String agentDefinitionStableKey,
        String agentName,
        String description,
        String systemPrompt,
        int maxIters,
        List<AgentKernelToolManifest> toolManifest,
        Set<String> toolWhitelist,
        String toolWhitelistVersion) {
    public AgentKernelSpec {
        toolManifest = List.copyOf(toolManifest);
        toolWhitelist = Set.copyOf(toolWhitelist);
        if (!key.agentDefinitionStableKey().equals(agentDefinitionStableKey)
                || !key.toolWhitelistVersion().equals(toolWhitelistVersion)) {
            throw new IllegalArgumentException("AgentKernelSpec key does not match definition/whitelist version");
        }
    }
}

public interface AgentKernelToolRegistry {
    AgentKernelToolkitResources register(AgentKernelSpec spec, Toolkit toolkit);
}

public interface AgentKernelToolkitResources extends AutoCloseable {
    static AgentKernelToolkitResources none() { return () -> { }; }
    @Override void close();
}
```

Build `toolManifestFingerprint` as SHA-256 over manifest entries sorted by `toolName`, with each canonical row encoded as `toolName|schemaSha256|readOnly|concurrencySafe`; build the whitelist version from a separately versioned, sorted whitelist. The key must never depend on mutable list/set iteration order. `AgentScopeRuntimeConfiguration` exposes a `@ConditionalOnMissingBean(AgentKernelToolRegistry.class)` no-op registry returning `AgentKernelToolkitResources.none()`, so later tool plans replace the bean without changing `AgentScopeHarnessFactory`.

- [ ] **Step 3: Make the existing model factory the single kernel-model bean**

```java
public interface OwnedChatModel extends AutoCloseable {
    static OwnedChatModel owned(ChatModelBase model) {
        return new DefaultOwnedChatModel(model);
    }
    ChatModelBase model();
    @Override void close();
}

public interface AgentKernelModelFactory {
    OwnedChatModel create(AgentKernelSpec spec);
}

@Component
@RequiredArgsConstructor
public final class AgentScopeModelFactory implements AgentKernelModelFactory {
    private final AiProviderService aiProviderService;

    @Override
    public OwnedChatModel create(AgentKernelSpec spec) {
        ChatModelBase model = aiProviderService.createAgentScopeModel(spec.model());
        return OwnedChatModel.owned(model);
    }
}
```

`DefaultOwnedChatModel` is a package-private final implementation in `OwnedChatModel.java`; it stores the non-null `ChatModelBase`, uses `AtomicBoolean.compareAndSet(false, true)`, and on first close invokes `close()` only when the model implements `AutoCloseable`. Remove `AgentScopeModelFactory.modelCache`, `getOrCreate`, `evict`, and `evictAll`. Remove `AgentScopeModelFactory` cache-invalidation injection/calls from `AiModelService` and `ApiConfigService`, then update `AiModelServiceTests` constructor fixtures.

Until the separate Pipeline cutover, `AgentScopeAssistantService` must not recreate a hidden cache. Replace its factory lookup with one per-stream `OwnedChatModel` obtained through the same `AgentScopeModelFactory.create(AgentKernelSpec)` contract, use `owned.model()` to build its transitional `ReActAgent`, and close the owner exactly once from complete/error/cancel cleanup. The spec it supplies uses the current agent/model/prompt fingerprints and the actual filtered tool manifest/whitelist; this keeps cache identity correct before Task 9 centralizes leasing.

- [ ] **Step 4: Build the minimal Harness through the registry**

```java
public AgentKernelResource create(AgentKernelSpec spec) {
    OwnedChatModel ownedModel = modelFactory.create(spec);
    Toolkit toolkit = new Toolkit();
    AgentKernelToolkitResources toolResources = toolRegistry.register(spec, toolkit);
    try {
        HarnessAgent agent = HarnessAgent.builder()
                .agentId(spec.agentDefinitionStableKey()).name(spec.agentName())
                .description(spec.description()).sysPrompt(spec.systemPrompt())
                .model(new StateStoreGuardedChatModel(ownedModel.model(), failures))
                .stateStore(stateStore).toolkit(toolkit).maxIters(spec.maxIters())
                .disableFilesystemTools().disableShellTool().disableMemoryTools().disableMemoryHooks()
                .disableSessionPersistence().disableWorkspaceContext().disableAtPathExpansion()
                .disableSubagents().disableDynamicSubagents().disableDefaultWorkspaceSkills()
                .disableToolsConfig().disableCompaction().disableToolResultEviction()
                .disableDynamicSkills().skillsEnabled(false).build();
        return new AgentKernelResource(agent, ownedModel, toolResources);
    } catch (RuntimeException failure) {
        toolResources.close();
        ownedModel.close();
        throw failure;
    }
}
```

Never retain a permanent `.toolkit(new Toolkit())` path that bypasses `AgentKernelToolRegistry`. `AgentKernelResource.close()` uses one `AtomicBoolean`; on first close it closes `HarnessAgent`, then registry-owned tool resources, then the owned model, combines suppressed failures, and does not close `AgentStateStore`.

- [ ] **Step 5: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentKernelResourceTests,AgentScopeHarnessFactoryTests,AgentKernelModelFactoryContextTests,AiModelServiceTests" test`

Expected: PASS; there is exactly one `AgentKernelModelFactory` bean, key identity includes manifest/whitelist versions, registry tools reach the Toolkit, and all owned resources close once while the Store remains open.

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeConfiguration.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeModelFactory.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/AiModelService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/ApiConfigService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/AiModelServiceTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel
git commit -m "feat: build minimal AgentScope kernel resource"
```

---

### Task 9: Enforce hard-cap leases and usingWhen execution

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/HarnessLease.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/HarnessLeaseCache.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentScopeHarnessInvoker.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/DefaultAgentScopeHarnessInvoker.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/message/AgentScopeMessageMapper.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRuntimeShutdownPort.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentScopeKernelLifecycle.java`
- Durable-owned documentation contract; do not create in this plan: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/StartAgentExecutionCommand.java`
- Durable-owned documentation contract; do not create in this plan: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ResumeAgentExecutionCommand.java`
- Durable-owned documentation contract; do not create in this plan: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ExecutionStopReason.java`
- Durable-owned documentation contract; do not create in this plan: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/RunExecutionSupervisor.java`
- Durable-owned documentation contract; do not create in this plan: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/CancellationCoordinator.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/HarnessLeaseCacheTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/DefaultAgentScopeHarnessInvokerTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/message/AgentScopeMessageMapperTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AgentRuntimeShutdownPortTests.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AgentScopeKernelLifecycleTests.java`

**Interfaces:**
- Produces: `Mono<HarnessLease> HarnessLeaseCache.acquire(AgentKernelSpec spec)` and `Mono<Void> HarnessLeaseCache.drainAndClose(Duration timeout)`.
- Produces: `Mono<Msg> AgentScopeHarnessInvoker.call(AgentKernelSpec spec, List<Msg> messages, RuntimeContext context)` and `Flux<AgentEvent> AgentScopeHarnessInvoker.streamEvents(AgentKernelSpec spec, List<Msg> messages, RuntimeContext context)`.
- Produces: kernel-owned `Mono<Void> AgentRuntimeShutdownPort.shutdown(Duration drainTimeout)`; the lifecycle depends only on this port and the lease cache.
- Documents the final durable-owned run orchestration signatures without creating or compiling those future production types in this kernel slice.

- [ ] **Step 1: Write red capacity/cleanup tests**

Cover 64 active keys, 65th 503 after exactly 5 seconds, same-key concurrent creation invokes the factory once, idle LRU eviction, acquire-before-subscribe cancel, complete/error/cancel, and double close. For shutdown, assert the cache rejects new acquires once draining starts, waits for active leases, closes each idle resource once, completes when the final lease returns, and on timeout reports `HARNESS_DRAIN_TIMEOUT` without evicting an active entry.

In `AgentRuntimeShutdownPortTests`, compile and reflect only `Mono<Void> shutdown(Duration drainTimeout)`. Walk `src/main/java` and assert `AgentRuntimeShutdownPort` and `AgentScopeKernelLifecycle` are under `com/stonewu/fusion/service/ai/run/`. Do not reference, reflect, or create the durable-owned start/resume/cancel records and interfaces in kernel tests.

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=HarnessLeaseCacheTests,DefaultAgentScopeHarnessInvokerTests,AgentScopeMessageMapperTests,AgentRuntimeShutdownPortTests,AgentScopeKernelLifecycleTests" test`

Expected: FAIL to compile because the cache, invoker, shutdown port, and lifecycle do not exist.

- [ ] **Step 2: Implement hard capacity and resource scope**

```java
Object keyLock = creationLocks.computeIfAbsent(spec.key(), ignored -> new Object());
try {
    synchronized (keyLock) {
        Entry existing = entries.get(spec.key());
        if (existing != null) return existing.acquireLease();
        evictOneExpiredOrLeastRecentlyUsedIdleEntry();
        if (!capacityPermits.tryAcquire()) {
            if (System.nanoTime() >= deadlineNanos) {
                return Mono.error(new BusinessException(503, "HARNESS_CAPACITY_EXHAUSTED"));
            }
            return Mono.delay(pollInterval).then(acquireUntil(spec, deadlineNanos));
        }
        try {
            Entry created = new Entry(factory.create(spec), capacityPermits::release);
            entries.put(spec.key(), created);
            return created.acquireLease();
        } catch (RuntimeException failure) {
            capacityPermits.release();
            throw failure;
        }
    }
} finally {
    creationLocks.remove(spec.key(), keyLock);
}
```

The semaphore starts with exactly `maximumSize` permits and a permit is returned only when an entry has zero leases and is actually removed/closed. The per-key lock performs a second lookup inside the critical section; different keys still create concurrently, while simultaneous misses for one key cannot build two Kernels. Run creation/acquire bookkeeping on the model scheduler, never on the Web event loop.

```java
return Flux.usingWhen(cache.acquire(spec),
        lease -> preflight.check(context).thenMany(lease.resource().agent().streamEvents(messages, context)),
        lease -> Mono.fromRunnable(lease::close),
        (lease, failure) -> Mono.fromRunnable(lease::close),
        lease -> Mono.fromRunnable(lease::close));
```

`call` uses five-argument `Mono.usingWhen`. Lease close uses `AtomicBoolean`. Strong text mapping returns `List.of(new UserMessage(List.of(TextBlock.builder().text(text).build())))`.

- [ ] **Step 3: Implement deterministic cache drain**

```java
public Mono<Void> drainAndClose(Duration timeout) {
    Mono<Void> existing = drainSignal.get();
    if (existing != null) return existing;
    Mono<Void> candidate = Mono.defer(() -> {
        draining.set(true);
        return awaitZeroActiveLeases()
                .timeout(timeout, Mono.error(
                        new BusinessException(503, "HARNESS_DRAIN_TIMEOUT")))
                .then(Mono.fromRunnable(this::closeAllIdleEntriesOnce));
    }).cache();
    return drainSignal.compareAndSet(null, candidate) ? candidate : drainSignal.get();
}
```

`drainSignal` is an `AtomicReference<Mono<Void>>`. The first subscription atomically changes the cache to draining; every later `acquire` fails with `BusinessException(503, "HARNESS_SHUTTING_DOWN")`. `awaitZeroActiveLeases` is signal-driven by lease releases, not a blocking loop. Timeout leaves active entries open and returns an error. Repeated drain calls share the same terminal result and never double-close resources.

- [ ] **Step 4: Implement the kernel shutdown port and document the durable boundary**

This interface is production code owned by this plan:

```java
public interface AgentRuntimeShutdownPort {
    Mono<Void> shutdown(Duration drainTimeout);
}
```

The following block freezes the signatures that the durable-runtime plan must create later. It is documentation only in this plan: do not create these records/interfaces, import their durable model types, or compile them in a kernel test.

```java
public record StartAgentExecutionCommand(
        StartedAgentRun run,
        List<Msg> messages,
        AgentKernelSnapshot kernelSnapshot,
        AgentKernelSpec kernelSpec,
        AgentScopeRuntimeContextRequest runtimeContextRequest) {}

public record ResumeAgentExecutionCommand(
        ResumedAgentRun run,
        List<Msg> messages,
        AgentKernelSnapshot kernelSnapshot,
        AgentScopeRuntimeContextRequest runtimeContextRequest) {}

public enum ExecutionStopReason {
    CANCEL_REQUESTED, OWNER_FENCED, DEADLINE, SHUTDOWN
}

public interface RunExecutionSupervisor extends AgentRuntimeShutdownPort {
    Mono<Void> start(StartAgentExecutionCommand command);
    Mono<Void> resume(ResumeAgentExecutionCommand command);
    Mono<Boolean> interruptOwned(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            ExecutionStopReason reason);
}

public interface CancellationCoordinator {
    Mono<AgentRunStatus> cancel(String runId, long currentUserId);
}
```

`StartedAgentRun`, `ResumedAgentRun`, `AgentKernelSnapshot`, `AgentRunStatus`, both commands, `ExecutionStopReason`, `RunExecutionSupervisor`, and `CancellationCoordinator` are owned and created by the durable-runtime plan. That plan makes `RunExecutionSupervisor extends AgentRuntimeShutdownPort`; it must not recreate the shutdown port. Its `start` and `resume` transfer ownership of the execution subscription before completing, and `interruptOwned` returns `true` only when the exact `(runId, ownerInstanceId, ownerEpoch)` owner was interrupted. Its inherited `shutdown` must atomically reject new `start`/`resume`, persist cancel or terminal intent for every active run, wait up to `drainTimeout` for those runs to release leases, then invoke `HarnessLeaseCache.drainAndClose(drainTimeout)`.

`AgentScopeKernelLifecycle` implements asynchronous Spring `SmartLifecycle.stop(Runnable)` and consumes `ObjectProvider<AgentRuntimeShutdownPort>`, never `RunExecutionSupervisor`. When a shutdown-port bean is present it awaits `port.shutdown(timeout)`; when none exists in this kernel-only phase it invokes `cache.drainAndClose(timeout)` directly. It invokes the callback on completion or error without `.block()`.

- [ ] **Step 5: Verify and commit**

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=HarnessLeaseCacheTests,DefaultAgentScopeHarnessInvokerTests,AgentScopeMessageMapperTests,AgentRuntimeShutdownPortTests,AgentScopeKernelLifecycleTests" test`

Expected: PASS; cache size never exceeds 64, same-key creation is single-flight, all `usingWhen` exits release once, drain is non-blocking and idempotent, the lifecycle works with and without `AgentRuntimeShutdownPort`, and kernel tests compile without any durable-owned type.

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/message ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRuntimeShutdownPort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentScopeKernelLifecycle.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/message ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AgentRuntimeShutdownPortTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AgentScopeKernelLifecycleTests.java
git commit -m "feat: lease AgentScope kernels with hard capacity"
```

---

### Task 10: Prove no-tool call/streamEvents and finish

**Files:**
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeNoToolKernelIntegrationTests.java`
- Modify: `ai-fusion-video/src/main/resources/application.yaml`
- Modify: `ai-fusion-video/src/main/resources/application-local.yaml`
- Modify: `ai-fusion-video/src/main/resources/application-docker.yaml`

**Interfaces:**
- Consumes: Tasks 3–9.
- Produces: no-tool `call`/`Flux<AgentEvent>` proof and environment configuration.

- [ ] **Step 1: Write the red integration test**

Use `EchoModel extends ChatModelBase`, an `AgentKernelSpec` with an empty manifest/whitelist and explicit whitelist version, two typed contexts, and StepVerifier. Assert `AgentScopeHarnessInvoker.call` returns assistant text; `streamEvents` contains `AGENT_START`, model/text events, and `AGENT_END`; the registry is still invoked for the empty-tool spec; and the two Store slots remain isolated.

Run: `.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeNoToolKernelIntegrationTests" test`

Expected: FAIL until all beans/configuration are wired.

- [ ] **Step 2: Bind exact environment values**

```yaml
fusion:
  agentscope:
    v2:
      cache:
        maximum-size: 64
        expire-after-access: 30m
        capacity-wait: 5s
      state:
        key-prefix: "afv:agentscope:v2:"
```

Local sets mode `in-memory`. Docker sets mode `redis` and `spring.data.redis.timeout: 2s`.

- [ ] **Step 3: Run final verification**

```powershell
.\\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeNoToolKernelIntegrationTests,AgentRuntimeShutdownPortTests" test
.\\mvnw.cmd dependency:tree "-Dincludes=io.agentscope"
.\\mvnw.cmd test
.\\mvnw.cmd package
$blocking = rg -n --glob '*.java' '\.block\(|\.toIterable\(|Thread\.sleep\(|ThreadLocal' src/main/java/com/stonewu/fusion/service/ai/agentscope src/main/java/com/stonewu/fusion/service/ai/run src/main/java/com/stonewu/fusion/service/ai/AgentConversationService.java src/main/java/com/stonewu/fusion/controller/ai/AiAssistantController.java src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeConfiguration.java src/main/java/com/stonewu/fusion/config/AgentScopeRuntimeProperties.java
if ($LASTEXITCODE -eq 0) { $blocking; throw 'forbidden blocking construct remains' }
if ($LASTEXITCODE -ne 1) { throw 'blocking source scan failed' }
$oldJava = rg -n --glob '*.java' --glob '!AgentScopeGaDependencyContractTests.java' 'io\.agentscope\.core\.(model\.(OpenAIChatModel|AnthropicChatModel|GeminiChatModel|DashScopeChatModel|OllamaChatModel)(\.Builder)?|formatter\.(gemini|anthropic)\.[A-Za-z0-9_$]+|session\.mysql\.[A-Za-z0-9_$]+)|AnthropicAgentScopeProxySupport|ProxyAwareAnthropicChatModel|GeminiToolResponseAwareChatFormatter|VertexAgentScopeProxySupport|MysqlSession' src/main src/test
if ($LASTEXITCODE -eq 0) { $oldJava; throw 'obsolete AgentScope V1 Java symbol remains' }
if ($LASTEXITCODE -ne 1) { throw 'AgentScope Java source scan failed' }
```

Expected: focused/full/package PASS; dependency tree contains only `2.0.0`; both scans return exit code 1 with no matches. Every specified-test command fails if either named test class is absent.

- [ ] **Step 4: Commit**

```powershell
git add ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeNoToolKernelIntegrationTests.java ai-fusion-video/src/main/resources/application.yaml ai-fusion-video/src/main/resources/application-local.yaml ai-fusion-video/src/main/resources/application-docker.yaml
git commit -m "test: prove minimal AgentScope V2 kernel"
```

---

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-dependency-kernel-implementation-plan.md`.

1. **Subagent-Driven (recommended):** use `superpowers:subagent-driven-development` with a fresh implementer and review gate per task.
2. **Inline Execution:** use `superpowers:executing-plans` in checkpointed batches.
