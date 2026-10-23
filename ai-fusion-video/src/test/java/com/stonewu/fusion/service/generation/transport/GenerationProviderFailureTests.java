package com.stonewu.fusion.service.generation.transport;

import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.sun.net.httpserver.HttpServer;
import okhttp3.MediaType;
import okhttp3.Request;
import okhttp3.RequestBody;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.io.ByteArrayOutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class GenerationProviderFailureTests {
    @ParameterizedTest
    @CsvSource({"400,false", "401,false", "403,false", "408,false", "429,false", "500,false", "503,false", "503,true"})
    void explicitHttpFailureBecomesDurableFailedReceiptWithoutResending(int status, boolean compressed) throws Exception {
        var calls = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/submit", exchange -> {
            calls.incrementAndGet();
            exchange.getRequestBody().readAllBytes();
            byte[] body = "{\"error\":{\"message\":\"video queue is full, please retry later\"}}".getBytes(StandardCharsets.UTF_8);
            if (compressed) {
                var bytes = new ByteArrayOutputStream();
                try (var gzip = new GZIPOutputStream(bytes)) { gzip.write(body); }
                body = bytes.toByteArray();
                exchange.getResponseHeaders().add("Content-Encoding", "gzip");
            }
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.getResponseHeaders().add("Retry-After", "0");
            exchange.sendResponseHeaders(status, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            var client = GenerationHttpClients.providerRequestBuilder().build();
            var request = new Request.Builder().url("http://127.0.0.1:" + server.getAddress().getPort() + "/submit")
                    .post(RequestBody.create("{}", MediaType.get("application/json"))).build();
            var prepared = new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
                try (var ignored = client.newCall(request).execute()) { throw new AssertionError("Error response must not be accepted"); }
                catch (IOException failure) { throw new IllegalStateException(failure); }
            });
            var receipt = prepared.submit();
            assertThat(receipt.result().state()).isEqualTo(RemoteGenerationResult.State.FAILED);
            assertThat(receipt.result().error()).contains("HTTP " + status, "video queue is full");
            assertThat(receipt.providerTaskId()).isNull();
            assertThat(calls).hasValue(1);
            assertThatThrownBy(prepared::submit).isInstanceOf(IllegalStateException.class);
            assertThat(calls).hasValue(1);
        } finally { server.stop(0); }
    }

    @Test void sdkCauseAndPreassignedRequestIdArePreservedForRejectedSubmission() {
        var prepared = new PreparedGenerationCall(new GenerationPreparedRequest("client-prompt", null), () -> {
            throw new IllegalStateException("SDK wrapper", GenerationProviderFailure.http(400, "{\"message\":\"Invalid workflow\"}"));
        });
        var receipt = prepared.submit();
        assertThat(receipt.providerTaskId()).isEqualTo("client-prompt");
        assertThat(receipt.result().state()).isEqualTo(RemoteGenerationResult.State.FAILED);
        assertThat(receipt.result().error()).contains("Invalid workflow");
    }

    @Test void missingResponseIsNotFabricatedAsAnExplicitProviderRejection() {
        var failure = new IllegalStateException("Connection interrupted", new IOException("No response"));
        var prepared = new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> { throw failure; });
        assertThatThrownBy(prepared::submit).isSameAs(failure);
    }

    @Test void errorReasonExcludesResponseStructureCredentialsAndHtml() {
        var denied = GenerationProviderFailure.http(401, "Invalid provider credential");
        assertThat(denied.getCode()).isEqualTo(502);
        assertThat(denied.httpStatus()).isEqualTo(401);
        assertThat(GenerationProviderFailure.http(401, "{\"error\":{\"message\":\"Invalid API key: sk-private-key-12345\"},\"debug\":\"private\"}").getMessage())
                .contains("HTTP 401", "Invalid").doesNotContain("private", "error\"", "debug", "sk-");
        assertThat(GenerationProviderFailure.http(503, "<!doctype html><html>debug details</html>").getMessage())
                .contains("HTTP 503", "错误页面").doesNotContain("debug", "<html");
    }
}
