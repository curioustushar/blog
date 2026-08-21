import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/blog/attention-evolved",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
