package com.stonewu.fusion.service.storage.strategy;

import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import com.stonewu.fusion.service.storage.StorageProviderRegistry;
import com.stonewu.fusion.service.storage.StorageUrlResolver;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class S3StorageStrategyTests {

    @Test
    void storeBytesUploadsWithResolvedConfigAndReturnsCustomDomainUrl() {
        S3ClientFactory factory = mock(S3ClientFactory.class);
        S3Client s3Client = mock(S3Client.class);
        when(factory.getClient(any())).thenReturn(s3Client);
        when(s3Client.putObject(any(PutObjectRequest.class), any(RequestBody.class)))
                .thenReturn(PutObjectResponse.builder().build());

        S3StorageStrategy strategy = new S3StorageStrategy(
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
                .basePath("/prefix/")
                .customDomain("https://cdn.example.com")
                .options("""
                        {"pathStyleAccessEnabled":true}
                        """)
                .build();

        String url = strategy.storeBytes("hello".getBytes(), "../images", "png", config);

        ArgumentCaptor<PutObjectRequest> requestCaptor = ArgumentCaptor.forClass(PutObjectRequest.class);
        verify(s3Client).putObject(requestCaptor.capture(), any(RequestBody.class));
        PutObjectRequest request = requestCaptor.getValue();
        assertThat(request.bucket()).isEqualTo("bucket");
        assertThat(request.key()).startsWith("prefix/images/");
        assertThat(request.key()).endsWith(".png");
        assertThat(request.contentType()).isEqualTo("image/png");
        assertThat(url).startsWith("https://cdn.example.com/prefix/images/");
        assertThat(url).endsWith(".png");
    }
}
