package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;
import okhttp3.Dns;
import okhttp3.HttpUrl;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.util.List;

@Component
public class MediaRemoteAddressPolicy {
    private final Dns dns;

    public MediaRemoteAddressPolicy() {
        this(Dns.SYSTEM);
    }

    MediaRemoteAddressPolicy(Dns dns) {
        this.dns = dns;
    }

    public record Target(URI uri, String host, List<InetAddress> addresses) {
        @Override
        public String toString() {
            return "MediaTarget[host=" + host + "]";
        }
    }

    public Target resolve(URI uri) throws UnknownHostException {
        String scheme = uri.getScheme();
        String host = uri.getHost();
        if ((!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) || host == null
                || uri.getRawUserInfo() != null || uri.getRawFragment() != null || uri.getRawAuthority().contains("%")
                || uri.getPort() == 0 || uri.getPort() > 65535) {
            throw denied();
        }
        String lookup = HttpUrl.get(uri.toString()).host();
        List<InetAddress> addresses = List.copyOf(dns.lookup(lookup));
        if (addresses.isEmpty() || addresses.stream().anyMatch(this::restricted)) {
            throw denied();
        }
        return new Target(uri, lookup, addresses);
    }

    private boolean restricted(InetAddress address) {
        if (address.isAnyLocalAddress() || address.isLoopbackAddress() || address.isLinkLocalAddress()
                || address.isSiteLocalAddress() || address.isMulticastAddress()) {
            return true;
        }
        byte[] bytes = address.getAddress();
        int a = bytes[0] & 255;
        int b = bytes[1] & 255;
        if (bytes.length == 4) {
            int c = bytes[2] & 255;
            return a == 0 || a == 10 || a == 127 || a >= 224 || (a == 100 && b >= 64 && b <= 127)
                    || (a == 169 && b == 254) || (a == 172 && b >= 16 && b <= 31)
                    || (a == 192 && b == 168) || (a == 192 && b == 0 && (c == 0 || c == 2))
                    || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
                    || (a == 203 && b == 0 && c == 113);
        }
        // 只接受全球单播，排除可能内嵌私网 IPv4 的过渡地址与文档地址。
        if ((a & 0xe0) != 0x20) {
            return true;
        }
        if (a == 0x20 && b == 0x02) {
            return true;
        }
        return a == 0x20 && b == 0x01 && ((bytes[2] == 0 && bytes[3] == 0)
                || ((bytes[2] & 255) == 0x0d && (bytes[3] & 255) == 0xb8));
    }

    private MediaException denied() {
        return new MediaException(400, "MEDIA_SOURCE_ADDRESS_DENIED", "媒体来源必须是可访问的公共 HTTP(S) 地址");
    }
}
