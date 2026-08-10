package com.stonewu.fusion.service.storage;

import com.stonewu.fusion.entity.storage.StorageConfig;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectResponse;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class StorageConnectionTestServiceTests {

    private HttpServer server;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void s3TestUploadsChecksPublicUrlAndDeletesHealthObject() throws IOException {
        String publicBaseUrl = startPublicServer(200);

        S3ClientFactory factory = mock(S3ClientFactory.class);
        S3Client s3Client = mock(S3Client.class);
        when(factory.getClient(any())).thenReturn(s3Client);
        when(s3Client.putObject(any(PutObjectRequest.class), any(RequestBody.class)))
                .thenReturn(PutObjectResponse.builder().build());
        when(s3Client.deleteObject(any(DeleteObjectRequest.class)))
                .thenReturn(DeleteObjectResponse.builder().build());

        StorageConnectionTestService service = new StorageConnectionTestService(
                new S3StorageConfigResolver(new StorageProviderRegistry()),
                factory,
                new StorageUrlResolver()
        );
        StorageConfig config = StorageConfig.builder()
                .type("s3")
                .provider("generic_s3")
                .endpoint("http://s3.local")
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .customDomain(publicBaseUrl)
                .options("""
                        {"pathStyleAccessEnabled":true}
                        """)
                .build();

        StorageConnectionTestResult result = service.test(config);

        assertThat(result.success()).isTrue();
        assertThat(result.publicUrl()).startsWith(publicBaseUrl + "/.healthcheck/");
        verify(s3Client).putObject(any(PutObjectRequest.class), any(RequestBody.class));
        verify(s3Client).deleteObject(any(DeleteObjectRequest.class));
    }

    @Test
    void s3TestExplainsPublicReadForbidden() throws IOException {
        String publicBaseUrl = startPublicServer(403);

        S3ClientFactory factory = mock(S3ClientFactory.class);
        S3Client s3Client = mock(S3Client.class);
        when(factory.getClient(any())).thenReturn(s3Client);
        when(s3Client.putObject(any(PutObjectRequest.class), any(RequestBody.class)))
                .thenReturn(PutObjectResponse.builder().build());
        when(s3Client.deleteObject(any(DeleteObjectRequest.class)))
                .thenReturn(DeleteObjectResponse.builder().build());

        StorageConnectionTestService service = new StorageConnectionTestService(
                new S3StorageConfigResolver(new StorageProviderRegistry()),
                factory,
                new StorageUrlResolver()
        );
        StorageConfig config = StorageConfig.builder()
                .type("s3")
                .provider("aliyun_oss")
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .region("cn-shanghai")
                .customDomain(publicBaseUrl)
                .build();

        StorageConnectionTestResult result = service.test(config);

        assertThat(result.success()).isFalse();
        assertThat(result.message())
                .contains("对象已上传")
                .contains("HTTP 403")
                .contains("允许匿名读取")
                .contains("签名 URL");
        verify(s3Client).deleteObject(any(DeleteObjectRequest.class));
    }

    private String startPublicServer(int statusCode) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            if ("HEAD".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(statusCode, -1);
            } else {
                byte[] body = "ok".getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(statusCode, body.length);
                exchange.getResponseBody().write(body);
            }
            exchange.close();
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }
}
