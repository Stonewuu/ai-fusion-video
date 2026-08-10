package com.stonewu.fusion.service.storage;

import com.stonewu.fusion.entity.storage.StorageConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class StorageUrlResolverTests {

    private final StorageProviderRegistry registry = new StorageProviderRegistry();
    private final StorageUrlResolver resolver = new StorageUrlResolver();

    @Test
    void customDomainTakesPrecedenceAndEncodesObjectKey() {
        ResolvedS3StorageConfig config = resolved("generic_s3", "https://s3.example.com", "us-east-1",
                "https://cdn.example.com", true);

        String url = resolver.resolvePublicUrl(config, "images/hello world/中文.png");

        assertThat(url).isEqualTo("https://cdn.example.com/images/hello%20world/%E4%B8%AD%E6%96%87.png");
    }

    @Test
    void aliyunUsesPublicOssTemplateWhenRegionIsAvailable() {
        ResolvedS3StorageConfig config = resolved("aliyun_oss", "https://s3.oss-cn-hangzhou.aliyuncs.com",
                "cn-hangzhou", null, false);

        String url = resolver.resolvePublicUrl(config, "videos/demo.mp4");

        assertThat(url).isEqualTo("https://bucket.oss-cn-hangzhou.aliyuncs.com/videos/demo.mp4");
    }

    @Test
    void genericPathStyleUsesEndpointBucketAndKey() {
        ResolvedS3StorageConfig config = resolved("generic_s3", "http://127.0.0.1:9000", "us-east-1",
                null, true);

        String url = resolver.resolvePublicUrl(config, "uploads/a.png");

        assertThat(url).isEqualTo("http://127.0.0.1:9000/bucket/uploads/a.png");
    }

    private ResolvedS3StorageConfig resolved(String provider, String endpoint, String region,
                                             String customDomain, boolean pathStyle) {
        StorageProviderProfile profile = registry.get(provider);
        StorageConfig source = StorageConfig.builder()
                .type("s3")
                .provider(provider)
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .build();
        return new ResolvedS3StorageConfig(
                source,
                profile,
                provider,
                endpoint,
                "bucket",
                "ak",
                "sk",
                region,
                region,
                null,
                customDomain,
                pathStyle,
                false,
                "WHEN_REQUIRED"
        );
    }
}
