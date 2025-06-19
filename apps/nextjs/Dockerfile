# apps/nextjs/Dockerfile

# Base image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy only pnpm files first for faster caching
COPY pnpm-lock.yaml ./
COPY package.json ./

# Install deps
RUN pnpm install

# Copy the rest of the app
COPY . .

# Build Next.js app
# RUN pnpm build

# Expose port
EXPOSE 3000

# Start the server
CMD ["sh", "-c", "pnpm run dev & while ! nc -z localhost 3000; do sleep 1; done; tail -f /dev/null"]
