package com.stonewu.fusion.service.storage;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class S3StorageConfigResolverTests {

    private final S3StorageConfigResolver resolver = new S3StorageConfigResolver(new StorageProviderRegistry());

    @Test
    void resolvesAliyunProfileEndpointAndDefaults() {
        StorageConfig config = StorageConfig.builder()
                .type("s3")
                .provider("aliyun_oss")
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .region("cn-hangzhou")
                .build();

        ResolvedS3StorageConfig resolved = resolver.resolve(config);

        assertThat(resolved.provider()).isEqualTo("aliyun_oss");
        assertThat(resolved.endpoint()).isEqualTo("https://s3.oss-cn-hangzhou.aliyuncs.com");
        assertThat(resolved.signingRegion()).isEqualTo("cn-hangzhou");
        assertThat(resolved.pathStyleAccessEnabled()).isFalse();
        assertThat(resolved.chunkedEncodingEnabled()).isFalse();
    }

    @Test
    void normalizesLegacyTencentTypeToProviderProfile() {
        StorageConfig config = StorageConfig.builder()
                .type("tencent_cos")
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .region("ap-shanghai")
                .build();

        ResolvedS3StorageConfig resolved = resolver.resolve(config);

        assertThat(resolved.provider()).isEqualTo("tencent_cos");
        assertThat(resolved.endpoint()).isEqualTo("https://cos.ap-shanghai.myqcloud.com");
        assertThat(resolved.pathStyleAccessEnabled()).isFalse();
    }

    @Test
    void appliesExplicitOptionsOverProfileDefaults() {
        StorageConfig config = StorageConfig.builder()
                .type("s3")
                .provider("aliyun_oss")
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .region("cn-hangzhou")
                .options("""
                        {"pathStyleAccessEnabled":true,"chunkedEncodingEnabled":true,"signingRegion":"oss-cn-hangzhou"}
                        """)
                .build();

        ResolvedS3StorageConfig resolved = resolver.resolve(config);

        assertThat(resolved.pathStyleAccessEnabled()).isTrue();
        assertThat(resolved.chunkedEncodingEnabled()).isTrue();
        assertThat(resolved.signingRegion()).isEqualTo("oss-cn-hangzhou");
    }

    @Test
    void requiresEndpointForGenericS3() {
        StorageConfig config = StorageConfig.builder()
                .type("s3")
                .provider("generic_s3")
                .bucketName("bucket")
                .accessKey("ak")
                .secretKey("sk")
                .build();

        assertThatThrownBy(() -> resolver.resolve(config))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("endpoint");
    }
}
