package com.stonewu.fusion.convert.storage;

import com.stonewu.fusion.controller.storage.vo.StorageConfigRespVO;
import com.stonewu.fusion.entity.storage.StorageConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class StorageConfigConvertTests {

    @Test
    void masksAccessKeyAndHidesSecretKey() {
        StorageConfig config = StorageConfig.builder()
                .name("oss")
                .type("s3")
                .provider("aliyun_oss")
                .endpoint("https://s3.oss-cn-hangzhou.aliyuncs.com")
                .bucketName("media-bucket")
                .accessKey("LTAI1234567890")
                .secretKey("secret")
                .options("""
                        {"pathStyleAccessEnabled":false}
                        """)
                .build();

        StorageConfigRespVO respVO = StorageConfigConvert.INSTANCE.convert(config);

        assertThat(respVO.getName()).isEqualTo("oss");
        assertThat(respVO.getType()).isEqualTo("s3");
        assertThat(respVO.getProvider()).isEqualTo("aliyun_oss");
        assertThat(respVO.getEndpoint()).isEqualTo("https://s3.oss-cn-hangzhou.aliyuncs.com");
        assertThat(respVO.getBucketName()).isEqualTo("media-bucket");
        assertThat(respVO.getAccessKey()).isEqualTo("LTAI****7890");
        assertThat(respVO.getSecretKey()).isNull();
        assertThat(respVO.getOptions()).containsEntry("pathStyleAccessEnabled", false);
    }
}
