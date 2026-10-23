package com.stonewu.fusion.service.generation.transport;

import okhttp3.*;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.client.ClientHttpRequest;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.ClientHttpResponse;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Spring SDK 与统一单次 OkHttp 传输间的适配，不依赖 Spring 已弃用的 OkHttp 工厂。
 */
final class GenerationHttpRequestFactory implements ClientHttpRequestFactory {
    private final OkHttpClient client;

    GenerationHttpRequestFactory(OkHttpClient client) {
        this.client = client;
    }

    @Override
    public ClientHttpRequest createRequest(URI uri, HttpMethod method) {
        return new ClientHttpRequest() {
            private final HttpHeaders headers = new HttpHeaders();
            private final ByteArrayOutputStream body = new ByteArrayOutputStream();
            private final Map<String, Object> attributes = new LinkedHashMap<>();
            private boolean executed;

            @Override
            public HttpMethod getMethod() {
                return method;
            }

            @Override
            public URI getURI() {
                return uri;
            }

            @Override
            public Map<String, Object> getAttributes() {
                return attributes;
            }

            @Override
            public HttpHeaders getHeaders() {
                return headers;
            }

            @Override
            public OutputStream getBody() {
                return body;
            }

            @Override
            public ClientHttpResponse execute() throws IOException {
                if (executed) {
                    throw new IllegalStateException("生成请求已执行，禁止再次发送");
                }
                executed = true;
                var builder = new Request.Builder().url(uri.toURL());
                headers.forEach((name, values) -> values.forEach(value -> builder.addHeader(name, value)));
                RequestBody requestBody = null;
                if (!method.equals(HttpMethod.GET) && !method.equals(HttpMethod.HEAD)) {
                    var contentType = headers.getContentType();
                    requestBody = RequestBody.create(body.toByteArray(),
                            contentType == null ? null : MediaType.get(contentType.toString()));
                }
                var response = client.newCall(builder.method(method.name(), requestBody).build()).execute();
                return new SpringResponse(response);
            }
        };
    }

    private record SpringResponse(Response response) implements ClientHttpResponse {
        @Override
        public HttpStatusCode getStatusCode() {
            return HttpStatusCode.valueOf(response.code());
        }

        @Override
        public String getStatusText() {
            return response.message();
        }

        @Override
        public HttpHeaders getHeaders() {
            var headers = new HttpHeaders();
            headers.putAll(response.headers().toMultimap());
            return HttpHeaders.readOnlyHttpHeaders(headers);
        }

        @Override
        public InputStream getBody() {
            return response.body() == null ? InputStream.nullInputStream() : response.body().byteStream();
        }

        @Override
        public void close() {
            response.close();
        }
    }
}
