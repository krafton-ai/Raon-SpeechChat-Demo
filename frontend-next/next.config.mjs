/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/fd-demo',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
