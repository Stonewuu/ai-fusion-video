import type { NextConfig } from "next";

const developmentBackendUrl = process.env.DEV_BACKEND_URL?.trim().replace(/\/+$/, "");

if (process.env.NODE_ENV === "development" && !developmentBackendUrl) {
  throw new Error(
    "本地开发必须配置 DEV_BACKEND_URL，请检查 ai-fusion-video-web/.env.development",
  );
}

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
  async rewrites() {
    if (process.env.NODE_ENV !== "development") {
      return [];
    }

    return [
      {
        source: "/api/:path*",
        destination: `${developmentBackendUrl!}/api/:path*`,
      },
      {
        source: "/media/:path*",
        destination: `${developmentBackendUrl!}/media/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: "/runtime-config.js",
        headers: [
          {
            key: "Cache-Control",
            value: "no-store, max-age=0",
          },
        ],
      },
    ];
  },
};

export default nextConfig;

