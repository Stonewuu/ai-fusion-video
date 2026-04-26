package com.stonewu.fusion.service.ai.proxy;

import cn.hutool.core.util.StrUtil;
import com.google.api.client.http.javanet.ConnectionFactory;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.auth.http.HttpTransportFactory;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.model.transport.HttpTransportConfig;
import io.agentscope.core.model.transport.OkHttpTransport;
import io.agentscope.core.model.transport.ProxyConfig;
import io.agentscope.core.model.transport.ProxyType;
import okhttp3.OkHttpClient;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;
import reactor.netty.transport.ProxyProvider;

import java.io.IOException;
import java.net.Authenticator;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.PasswordAuthentication;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.URL;
import java.net.http.HttpClient.Builder;
import java.time.Duration;
import java.util.Locale;

/**
 * AI 提供商出站代理工具。
 * <p>
 * 代理配置挂在 {@link ApiConfig} 上，便于按不同供应商使用不同线路。
 */
public final class AiProxySupport {

    public static final String TYPE_NONE = "none";
    public static final String TYPE_HTTP = "http";
    public static final String TYPE_SOCKS5 = "socks5";

    private AiProxySupport() {
    }

    public static String normalizeProxyType(String proxyType) {
        if (StrUtil.isBlank(proxyType)) {
            return TYPE_NONE;
        }
        String normalized = proxyType.trim().toLowerCase(Locale.ROOT).replace("_", "").replace("-", "");
        return switch (normalized) {
            case "http", "https" -> TYPE_HTTP;
            case "socks", "socks5" -> TYPE_SOCKS5;
            case "none", "off", "disabled" -> TYPE_NONE;
            default -> normalized;
        };
    }

    public static boolean isSupportedProxyType(String proxyType) {
        String normalized = normalizeProxyType(proxyType);
        return TYPE_HTTP.equals(normalized) || TYPE_SOCKS5.equals(normalized);
    }

    public static boolean isEnabled(ApiConfig apiConfig) {
        return resolveNullable(apiConfig) != null;
    }

    public static Proxy javaProxyOrNull(ApiConfig apiConfig) {
        ProxySettings settings = resolveNullable(apiConfig);
        return settings == null ? null : settings.toJavaProxy();
    }

    public static OkHttpClient.Builder applyToOkHttp(OkHttpClient.Builder builder, ApiConfig apiConfig) {
        ProxySettings settings = resolveNullable(apiConfig);
        if (settings != null) {
            builder.proxy(settings.toJavaProxy());
            if (settings.hasHttpCredentials()) {
                builder.proxyAuthenticator((route, response) -> {
                    if (response.request().header("Proxy-Authorization") != null) {
                        return null;
                    }
                    return response.request().newBuilder()
                            .header("Proxy-Authorization", settings.basicProxyAuthorization())
                            .build();
                });
            }
        }
        return builder;
    }

    public static OkHttpClient okHttpClient(OkHttpClient baseClient, ApiConfig apiConfig) {
        if (!isEnabled(apiConfig)) {
            return baseClient;
        }
        return applyToOkHttp(baseClient.newBuilder(), apiConfig).build();
    }

    public static ClientHttpRequestFactory createRequestFactory(ApiConfig apiConfig,
                                                                int connectTimeoutMillis,
                                                                int readTimeoutMillis) {
        ProxySettings settings = resolveNullable(apiConfig);
        if (settings != null && settings.hasHttpCredentials()) {
            return createJdkRequestFactory(settings, connectTimeoutMillis, readTimeoutMillis);
        }
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(connectTimeoutMillis);
        requestFactory.setReadTimeout(readTimeoutMillis);
        Proxy proxy = settings == null ? null : settings.toJavaProxy();
        if (proxy != null) {
            requestFactory.setProxy(proxy);
        }
        return requestFactory;
    }

    public static RestClient.Builder restClientBuilder(ApiConfig apiConfig,
                                                       int connectTimeoutMillis,
                                                       int readTimeoutMillis) {
        return RestClient.builder().requestFactory(
                createRequestFactory(apiConfig, connectTimeoutMillis, readTimeoutMillis));
    }

    public static WebClient.Builder webClientBuilder(ApiConfig apiConfig,
                                                     String connectionProviderName,
                                                     Duration responseTimeout) {
        ConnectionProvider provider = ConnectionProvider.builder(connectionProviderName)
                .maxConnections(500)
                .maxIdleTime(Duration.ofSeconds(45))
                .maxLifeTime(Duration.ofMinutes(10))
                .pendingAcquireTimeout(Duration.ofSeconds(60))
                .evictInBackground(Duration.ofSeconds(30))
                .build();
        HttpClient httpClient = HttpClient.create(provider)
                .compress(true)
                .keepAlive(true)
                .responseTimeout(responseTimeout);
        httpClient = applyToReactorNetty(httpClient, apiConfig);
        return WebClient.builder().clientConnector(new ReactorClientHttpConnector(httpClient));
    }

    public static HttpClient applyToReactorNetty(HttpClient httpClient, ApiConfig apiConfig) {
        ProxySettings settings = resolveNullable(apiConfig);
        if (settings == null) {
            return httpClient;
        }
        return httpClient.proxy(proxy -> {
            ProxyProvider.Builder builder = proxy
                    .type(settings.toReactorProxyType())
                    .host(settings.host())
                    .port(settings.port());
            if (settings.hasCredentials()) {
                builder.username(settings.username());
                builder.password(ignored -> settings.passwordOrEmpty());
            }
        });
    }

    public static HttpTransportFactory googleHttpTransportFactory(ApiConfig apiConfig) {
        ProxySettings settings = resolveNullable(apiConfig);
        if (settings == null) {
            return null;
        }
        return () -> googleHttpTransport(settings);
    }

    public static com.google.api.client.http.HttpTransport googleHttpTransport(ApiConfig apiConfig) {
        return googleHttpTransport(resolveNullable(apiConfig));
    }

    public static ProxyConfig agentScopeProxyConfig(ApiConfig apiConfig) {
        ProxySettings settings = resolveNullable(apiConfig);
        if (settings == null) {
            return null;
        }
        ProxyConfig.Builder proxyBuilder = ProxyConfig.builder()
                .type(settings.toAgentScopeProxyType())
                .host(settings.host())
                .port(settings.port());
        if (settings.hasCredentials()) {
            proxyBuilder.username(settings.username());
            proxyBuilder.password(settings.passwordOrEmpty());
        }
        return proxyBuilder.build();
    }

    public static io.agentscope.core.model.transport.HttpTransport agentScopeHttpTransport(ApiConfig apiConfig) {
        ProxyConfig proxyConfig = agentScopeProxyConfig(apiConfig);
        if (proxyConfig == null) {
            return null;
        }
        HttpTransportConfig transportConfig = HttpTransportConfig.builder()
                .connectTimeout(Duration.ofSeconds(60))
                .readTimeout(Duration.ofMinutes(3))
                .writeTimeout(Duration.ofSeconds(60))
                .proxy(proxyConfig)
                .build();
        return OkHttpTransport.builder().config(transportConfig).build();
    }

    private static com.google.api.client.http.HttpTransport googleHttpTransport(ProxySettings settings) {
        NetHttpTransport.Builder builder = new NetHttpTransport.Builder();
        if (settings != null) {
            if (settings.hasHttpCredentials()) {
                builder.setConnectionFactory(new ProxyConnectionFactory(settings));
            } else {
                builder.setProxy(settings.toJavaProxy());
            }
        }
        return builder.build();
    }

    private static ClientHttpRequestFactory createJdkRequestFactory(ProxySettings settings,
                                                                    int connectTimeoutMillis,
                                                                    int readTimeoutMillis) {
        Builder httpClientBuilder = java.net.http.HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(connectTimeoutMillis))
                .proxy(ProxySelector.of(new InetSocketAddress(settings.host(), settings.port())));
        if (settings.hasCredentials()) {
            httpClientBuilder.authenticator(new Authenticator() {
                @Override
                protected PasswordAuthentication getPasswordAuthentication() {
                    if (getRequestorType() != RequestorType.PROXY) {
                        return null;
                    }
                    return new PasswordAuthentication(settings.username(), settings.passwordOrEmpty().toCharArray());
                }
            });
        }
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClientBuilder.build());
        requestFactory.setReadTimeout(readTimeoutMillis);
        return requestFactory;
    }

    private static ProxySettings resolveNullable(ApiConfig apiConfig) {
        if (apiConfig == null) {
            return null;
        }
        String type = normalizeProxyType(apiConfig.getProxyType());
        if (TYPE_NONE.equals(type)) {
            return null;
        }
        if (!isSupportedProxyType(type)) {
            throw new BusinessException("不支持的代理类型: " + apiConfig.getProxyType());
        }
        if (StrUtil.isBlank(apiConfig.getProxyHost())) {
            throw new BusinessException("启用代理时代理主机不能为空");
        }
        Integer port = apiConfig.getProxyPort();
        if (port == null || port <= 0 || port > 65535) {
            throw new BusinessException("启用代理时代理端口必须在 1-65535 之间");
        }
        return new ProxySettings(type, apiConfig.getProxyHost().trim(), port,
                StrUtil.trim(apiConfig.getProxyUsername()), apiConfig.getProxyPassword());
    }

    private record ProxySettings(String type, String host, int port, String username, String password) {

        private boolean hasCredentials() {
            return StrUtil.isNotBlank(username);
        }

        private boolean hasHttpCredentials() {
            return TYPE_HTTP.equals(type) && hasCredentials();
        }

        private String passwordOrEmpty() {
            return password == null ? "" : password;
        }

        private String basicProxyAuthorization() {
            String credentials = username + ":" + passwordOrEmpty();
            return "Basic " + java.util.Base64.getEncoder().encodeToString(credentials.getBytes(java.nio.charset.StandardCharsets.ISO_8859_1));
        }

        private Proxy toJavaProxy() {
            Proxy.Type javaType = TYPE_HTTP.equals(type) ? Proxy.Type.HTTP : Proxy.Type.SOCKS;
            return new Proxy(javaType, new InetSocketAddress(host, port));
        }

        private ProxyProvider.Proxy toReactorProxyType() {
            return TYPE_HTTP.equals(type) ? ProxyProvider.Proxy.HTTP : ProxyProvider.Proxy.SOCKS5;
        }

        private ProxyType toAgentScopeProxyType() {
            return TYPE_HTTP.equals(type) ? ProxyType.HTTP : ProxyType.SOCKS5;
        }
    }

    private static final class ProxyConnectionFactory implements ConnectionFactory {

        private final ProxySettings settings;

        private ProxyConnectionFactory(ProxySettings settings) {
            this.settings = settings;
        }

        @Override
        public HttpURLConnection openConnection(URL url) throws IOException, ClassCastException {
            HttpURLConnection connection = (HttpURLConnection) url.openConnection(settings.toJavaProxy());
            if (settings.hasHttpCredentials()) {
                connection.setRequestProperty("Proxy-Authorization", settings.basicProxyAuthorization());
            }
            return connection;
        }
    }
}
