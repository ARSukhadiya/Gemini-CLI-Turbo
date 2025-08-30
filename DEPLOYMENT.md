# 🚀 Deployment Guide - Gemini CLI

> **Production-ready deployment guide for high-performance Gemini CLI systems**

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Performance Tuning](#performance-tuning)
5. [Monitoring & Observability](#monitoring--observability)
6. [Scaling Strategies](#scaling-strategies)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## 🎯 Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS 12+, Windows 10+
- **Python**: 3.9+ with pip
- **Memory**: Minimum 2GB RAM, Recommended 8GB+
- **Storage**: 1GB free space
- **Network**: Stable internet connection for API access

### Dependencies

- **Redis** (optional, for distributed caching)
- **MongoDB** (optional, for persistent caching)
- **Prometheus** (optional, for metrics collection)
- **Grafana** (optional, for metrics visualization)

## 🚀 Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/gemini-cli.git
cd gemini-cli

# Install with all dependencies
pip install -e .[all]

# Or install specific components
pip install -e .[performance,monitoring]
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env

# Set your API key
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Test Installation

```bash
# Check CLI works
gemini-cli --version

# Run interactive mode
gemini-cli --interactive

# Test performance
gemini-cli --benchmark --iterations 5
```

## 🏭 Production Deployment

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 gemini && chown -R gemini:gemini /app
USER gemini

# Expose metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gemini-cli --status || exit 1

# Default command
CMD ["gemini-cli", "--interactive"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  gemini-cli:
    build: .
    container_name: gemini-cli
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - METRICS_ENABLED=true
      - METRICS_PORT=9090
      - CACHE_ENABLED=true
      - REDIS_ENABLED=true
    ports:
      - "9090:9090"  # Metrics
    volumes:
      - ./cache:/app/cache
      - ./logs:/app/logs
    depends_on:
      - redis
      - prometheus
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "gemini-cli", "--status"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: gemini-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Kubernetes Deployment

#### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gemini-cli-config
data:
  GEMINI_MODEL: "gemini-pro"
  MAX_CONCURRENT_REQUESTS: "20"
  REQUEST_TIMEOUT: "30"
  CACHE_ENABLED: "true"
  METRICS_ENABLED: "true"
  METRICS_PORT: "9090"
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gemini-cli
  labels:
    app: gemini-cli
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gemini-cli
  template:
    metadata:
      labels:
        app: gemini-cli
    spec:
      containers:
      - name: gemini-cli
        image: your-registry/gemini-cli:latest
        ports:
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: gemini-cli-config
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gemini-api-secret
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gemini-cli-service
  labels:
    app: gemini-cli
spec:
  selector:
    app: gemini-cli
  ports:
  - port: 80
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
```

## ⚡ Performance Tuning

### System-Level Optimization

#### Linux Kernel Tuning

```bash
# /etc/sysctl.conf
# Network optimization
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.tcp_max_tw_buckets = 2000000

# File descriptor limits
fs.file-max = 2097152
fs.nr_open = 2097152

# Memory optimization
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
```

#### User Limits

```bash
# /etc/security/limits.conf
gemini soft nofile 65536
gemini hard nofile 65536
gemini soft nproc 32768
gemini hard nproc 32768
```

### Application-Level Optimization

#### Configuration Tuning

```bash
# .env
# Connection pooling
CONNECTION_POOL_SIZE=50
KEEP_ALIVE_TIMEOUT=120

# Concurrency
MAX_CONCURRENT_REQUESTS=25
BATCH_SIZE=10

# Caching
CACHE_MAX_SIZE=5000
CACHE_TTL=7200

# Performance
PREFETCH_ENABLED=true
ENABLE_COMPRESSION=true
```

#### Memory Optimization

```bash
# Enable garbage collection
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Use uvloop for better performance
export EVENT_LOOP_POLICY=uvloop
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "gemini_rules.yml"

scrape_configs:
  - job_name: 'gemini-cli'
    static_configs:
      - targets: ['gemini-cli:9090']
    metrics_path: /metrics
    scrape_interval: 5s
```

### Grafana Dashboards

#### Performance Dashboard

```json
{
  "dashboard": {
    "title": "Gemini CLI Performance",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gemini_requests_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Response Latency",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(gemini_request_duration_seconds_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(gemini_cache_hits_total[5m]) / rate(gemini_requests_total[5m])",
            "legendFormat": "Hit Rate"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# gemini_rules.yml
groups:
  - name: gemini-cli
    rules:
      - alert: HighErrorRate
        expr: rate(gemini_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(gemini_request_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: CircuitBreakerOpen
        expr: gemini_circuit_breaker_state == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker is open"
          description: "Circuit breaker has opened due to high failure rate"
```

## 📈 Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream gemini_cli {
    least_conn;
    server gemini-cli-1:9090 max_fails=3 fail_timeout=30s;
    server gemini-cli-2:9090 max_fails=3 fail_timeout=30s;
    server gemini-cli-3:9090 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name gemini.example.com;

    location / {
        proxy_pass http://gemini_cli;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

#### Auto-scaling with HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gemini-cli-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gemini-cli
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

#### Resource Optimization

```yaml
# Resource requests and limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🔒 Security Considerations

### API Key Management

```bash
# Use Kubernetes secrets
kubectl create secret generic gemini-api-secret \
  --from-literal=api-key="your-api-key-here"

# Or use external secret management
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: gemini-api-secret
spec:
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: gemini-api-secret
  data:
  - secretKey: api-key
    remoteRef:
      key: gemini/api-key
      property: value
```

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gemini-cli-network-policy
spec:
  podSelector:
    matchLabels:
      app: gemini-cli
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

## 🐛 Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
gemini-cli --stats

# Enable garbage collection
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Reduce cache size
CACHE_MAX_SIZE=1000
```

#### Connection Issues

```bash
# Check connection pool
gemini-cli --status

# Increase connection limits
CONNECTION_POOL_SIZE=100
KEEP_ALIVE_TIMEOUT=300

# Check network connectivity
curl -v https://generativelanguage.googleapis.com/health
```

#### Performance Degradation

```bash
# Run performance benchmark
gemini-cli --benchmark --iterations 100

# Check metrics
curl http://localhost:9090/metrics

# Analyze logs
tail -f logs/gemini-cli.log | grep "ERROR\|WARNING"
```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Enable verbose output
gemini-cli --log-level DEBUG --interactive

# Check configuration
gemini-cli --status --verbose
```

### Health Checks

```bash
# Basic health check
gemini-cli --status

# Detailed health check
curl http://localhost:9090/health

# Circuit breaker status
gemini-cli --stats | grep circuit_breaker
```

## 📚 Additional Resources

- [Performance Tuning Guide](PERFORMANCE.md)
- [API Reference](API.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/gemini-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gemini-cli/discussions)
- **Documentation**: [GitHub Wiki](https://github.com/yourusername/gemini-cli/wiki)

---

**Happy deploying! 🚀**

