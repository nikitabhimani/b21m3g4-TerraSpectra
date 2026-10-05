# syntax=docker/dockerfile:1.7
# TerraSpectra GIS 3D Dashboard. Build context: repository root.
#   docker build -f dashboard/Dockerfile -t terraspectra-dashboard .
#
# Stage 1: Build static bundle with Node 22
FROM node:22-alpine AS builder

WORKDIR /app/dashboard

# Copy package manifests first for optimal layer caching
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm ci

# Copy dashboard source code
COPY dashboard/ ./

# Build arguments for frontend configuration
ARG VITE_API_URL=http://localhost:8000
ARG VITE_API_KEY=dev-key-change-me
ENV VITE_API_URL=${VITE_API_URL} \
    VITE_API_KEY=${VITE_API_KEY}

RUN npm run build

# Stage 2: Serve via Nginx Alpine
FROM nginx:alpine AS runner

# Remove default nginx static html
RUN rm -rf /usr/share/nginx/html/*

# Copy custom Nginx configuration
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled assets from builder
COPY --from=builder /app/dashboard/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
