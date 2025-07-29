# apps/nextjs/Dockerfile

# Base image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm


COPY pnpm-lock.yaml ./
COPY package.json ./

RUN pnpm install

COPY . .


EXPOSE 3000

# Start the server
CMD ["sh", "-c", "pnpm run dev & while ! nc -z localhost 3000; do sleep 1; done; tail -f /dev/null"]
