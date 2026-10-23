package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.storage.LegacyMediaRoot;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Protocol;
import okhttp3.Response;
import okhttp3.ResponseBody;
import okio.Buffer;
import okio.BufferedSource;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.InetAddress;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MediaSourceFetcherTests {
    @TempDir Path directory;

    @Test
    void refusesPrivateMixedAndTransitionAddressesBeforeAnyHttpRequest() throws Exception {
        for (String address : List.of("127.0.0.1", "10.1.2.3", "172.16.1.2", "192.168.1.3", "169.254.169.254",
                "100.64.0.1", "198.18.1.2", "224.0.0.1", "::1", "fc00::1", "fe80::1", "64:ff9b::7f00:1", "2002:7f00:1::")) {
            var policy = new MediaRemoteAddressPolicy(host -> List.of(InetAddress.getByName(address)));
            assertThatThrownBy(() -> policy.resolve(URI.create("https://media.example/content"))).isInstanceOf(MediaException.class);
        }
        var mixed = new MediaRemoteAddressPolicy(host -> List.of(InetAddress.getByName("93.184.216.34"), InetAddress.getByName("127.0.0.1")));
        assertThatThrownBy(() -> mixed.resolve(URI.create("https://media.example/content"))).isInstanceOf(MediaException.class);
        var publicDns = new MediaRemoteAddressPolicy(host -> List.of(InetAddress.getByName("93.184.216.34")));
        for (String invalid : List.of("file:///etc/passwd", "ftp://public.example/data", "https://user:password@public.example/data",
                "http://public.example:0/data", "http://public.example:70000/data", "http://[fe80::1%25eth0]/data")) {
            assertThatThrownBy(() -> publicDns.resolve(URI.create(invalid))).isInstanceOf(MediaException.class);
        }
        assertThat(publicDns.resolve(URI.create("https://PUBLIC.example/path?signature=secret")).host()).isEqualTo("public.example");
        assertThat(publicDns.resolve(URI.create("https://public.example/path?signature=secret")).toString()).doesNotContain("signature", "secret");
    }

    @Test
    void acceptsBoundedPublicContentAndRevalidatesRedirectTargets() throws Exception {
        var root = mock(LegacyMediaRoot.class);
        var policy = new MediaRemoteAddressPolicy(host -> List.of(InetAddress.getByName(
                host.equals("public.example") ? "93.184.216.34" : "127.0.0.1")));
        AtomicInteger requests = new AtomicInteger();
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(chain -> {
            requests.incrementAndGet();
            assertThat(chain.request().header("Authorization")).isNull();
            assertThat(chain.request().header("Cookie")).isNull();
            if (chain.request().url().encodedPath().equals("/redirect")) {
                return new Response.Builder().request(chain.request()).protocol(Protocol.HTTP_1_1)
                        .code(302).message("redirect").header("Location", "http://private.example/admin")
                        .body(ResponseBody.create(new byte[0], MediaType.get("application/octet-stream"))).build();
            }
            return new Response.Builder().request(chain.request()).protocol(Protocol.HTTP_1_1).code(200).message("ok")
                    .body(ResponseBody.create(new byte[]{1, 5, 9}, MediaType.get("application/octet-stream"))).build();
        }).build();
        var fetcher = new MediaSourceFetcher(root, policy, client);
        Path target = directory.resolve("file");
        fetcher.copy("https://public.example/file", target);
        assertThat(Files.readAllBytes(target)).containsExactly((byte) 1, (byte) 5, (byte) 9);
        assertThatThrownBy(() -> fetcher.copy("https://public.example/redirect", target)).isInstanceOf(MediaException.class);
        assertThat(requests).hasValue(2);
    }

    @Test
    void rejectsRedirectLoopsAndOversizedResponsesWithoutReadingBody() throws Exception {
        var root = mock(LegacyMediaRoot.class);
        var policy = new MediaRemoteAddressPolicy(host -> List.of(InetAddress.getByName("93.184.216.34")));
        AtomicInteger requests = new AtomicInteger();
        var client = new OkHttpClient.Builder().addInterceptor(chain -> {
            requests.incrementAndGet();
            return new Response.Builder().request(chain.request()).protocol(Protocol.HTTP_1_1).code(302).message("redirect")
                    .header("Location", "/loop").body(ResponseBody.create(new byte[0], null)).build();
        }).build();
        var fetcher = new MediaSourceFetcher(root, policy, client);
        assertThatThrownBy(() -> fetcher.copy("https://public.example/loop", directory.resolve("loop"))).hasMessageContaining("重定向");
        assertThat(requests).hasValue(6);
        var oversized = new OkHttpClient.Builder().addInterceptor(chain -> new Response.Builder()
                .request(chain.request()).protocol(Protocol.HTTP_1_1).code(200).message("ok")
                .body(new ResponseBody() {
                    @Override public MediaType contentType() { return MediaType.get("application/octet-stream"); }
                    @Override public long contentLength() { return MediaIntakeService.MAX_RESULT_BYTES + 1; }
                    @Override public BufferedSource source() { return new Buffer(); }
                }).build()).build();
        Path output = directory.resolve("oversized");
        assertThatThrownBy(() -> new MediaSourceFetcher(root, policy, oversized).copy("https://public.example/large", output))
                .isInstanceOf(MediaException.class).hasMessageContaining("2 GiB");
        assertThat(Files.exists(output)).isFalse();
    }

    @Test
    void localImportsUseOnlyThePublishedRootAndRejectTraversalAndCiphertextObjects() throws Exception {
        var root = mock(LegacyMediaRoot.class);
        when(root.path()).thenReturn(directory);
        var fetcher = new MediaSourceFetcher(root, new MediaRemoteAddressPolicy());
        Files.write(directory.resolve("真实素材.png"), new byte[]{7, 8, 9});
        Path output = directory.resolve("received");
        fetcher.copy("/media/%E7%9C%9F%E5%AE%9E%E7%B4%A0%E6%9D%90.png", output);
        assertThat(Files.readAllBytes(output)).containsExactly((byte) 7, (byte) 8, (byte) 9);
        for (String uri : List.of("/media/../private.txt", "/media/%2e%2e/private.txt", "/media/a%5cb",
                "/media/a:stream", "/media/.afv-managed/private.afvblob", "//private.example/media/a.png", "/other/a.png",
                "/media/真实素材.png?key=anything")) {
            assertThatThrownBy(() -> fetcher.copy(uri, output)).isInstanceOf(MediaException.class);
        }
    }
}
