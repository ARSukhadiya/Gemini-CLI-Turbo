# 🚀 Running Instructions - Gemini CLI

> **Complete step-by-step guide to get Gemini CLI up and running**

## 📋 Quick Start (5 minutes)

### 1. **Prerequisites Check**
```bash
# Check Python version (3.9+ required)
python --version

# Check pip availability
pip --version

# Check git availability
git --version
```

### 2. **Clone and Install**
```bash
# Clone the repository
git clone https://github.com/yourusername/gemini-cli.git
cd gemini-cli

# Install with all dependencies
pip install -e .[all]

# Verify installation
gemini-cli --version
```

### 3. **Configure API Key**
```bash
# Copy environment template
cp env.example .env

# Edit configuration (replace with your actual API key)
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

### 4. **Test the CLI**
```bash
# Interactive mode
gemini-cli --interactive

# Or direct query
gemini-cli "Explain quantum computing in simple terms"
```

## 🏗️ Detailed Setup Instructions

### **Option A: Development Installation**

#### 1. **Clone Repository**
```bash
git clone https://github.com/yourusername/gemini-cli.git
cd gemini-cli
```

#### 2. **Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
# Install base package
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Install performance dependencies
pip install -e .[performance]

# Install monitoring dependencies
pip install -e .[monitoring]

# Install benchmarking dependencies
pip install -e .[benchmarking]

# Or install everything at once
pip install -e .[all]
```

#### 4. **Setup Pre-commit Hooks**
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### **Option B: Production Installation**

#### 1. **System-wide Installation**
```bash
# Install system-wide
sudo pip install -e .[performance,monitoring]

# Or using package manager
# On Ubuntu/Debian:
sudo apt-get install python3-pip
sudo pip3 install -e .[performance,monitoring]

# On macOS:
brew install python3
pip3 install -e .[performance,monitoring]
```

#### 2. **Docker Installation**
```bash
# Build Docker image
docker build -t gemini-cli .

# Run container
docker run -it --env-file .env gemini-cli
```

## ⚙️ Configuration

### **Environment Variables**

#### **Required Configuration**
```bash
# .env file
GEMINI_API_KEY=your_actual_api_key_here
```

#### **Performance Configuration**
```bash
# Connection and concurrency
MAX_CONCURRENT_REQUESTS=25
CONNECTION_POOL_SIZE=50
REQUEST_TIMEOUT=30
KEEP_ALIVE_TIMEOUT=120

# Caching
CACHE_ENABLED=true
CACHE_MAX_SIZE=5000
CACHE_TTL=7200
CACHE_STRATEGY=lru

# Optimization
BATCH_SIZE=10
PREFETCH_ENABLED=true
ENABLE_COMPRESSION=true
```

#### **Monitoring Configuration**
```bash
# Metrics and monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
LOG_LEVEL=INFO

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD=0.5
CIRCUIT_BREAKER_TIMEOUT=60
```

### **Configuration Files**

#### **Advanced Configuration**
```bash
# Create custom config file
mkdir -p config
cat > config/custom.yml << EOF
gemini:
  model: gemini-pro
  max_concurrent_requests: 50
  request_timeout: 60
  
performance:
  batch_size: 20
  prefetch_enabled: true
  
cache:
  strategy: hybrid
  max_size: 10000
  ttl: 14400
EOF
```

## 🚀 Running the CLI

### **Basic Usage**

#### **Interactive Mode**
```bash
# Start interactive CLI
gemini-cli --interactive

# Available commands in interactive mode:
# help     - Show help
# status   - Show system status
# stats    - Show performance statistics
# clear    - Clear console
# quit     - Exit CLI
```

#### **Direct Query Mode**
```bash
# Single query
gemini-cli "What is machine learning?"

# Streaming response
gemini-cli --stream "Write a Python function for binary search"

# With specific model
gemini-cli --model gemini-pro "Explain quantum computing"
```

#### **Batch Processing**
```bash
# Process multiple prompts
echo "prompt1\nprompt2\nprompt3" > prompts.txt
gemini-cli --batch prompts.txt

# Or use command line
gemini-cli --batch "prompt1" "prompt2" "prompt3"
```

### **Advanced Usage**

#### **Performance Benchmarking**
```bash
# Run performance benchmark
gemini-cli --benchmark --iterations 100

# Custom benchmark scenarios
gemini-cli --benchmark --scenarios all --duration 300

# Load testing
gemini-cli --load-test --users 50 --duration 600
```

#### **Monitoring and Metrics**
```bash
# Show real-time status
gemini-cli --status

# Show performance statistics
gemini-cli --stats

# Export metrics
gemini-cli --export-metrics prometheus

# Health check
gemini-cli --health
```

#### **Configuration Management**
```bash
# Show current configuration
gemini-cli --config show

# Validate configuration
gemini-cli --config validate

# Reload configuration
gemini-cli --config reload

# Export configuration
gemini-cli --config export
```

## 📊 Performance Testing

### **Quick Performance Test**
```bash
# Basic performance test
gemini-cli --benchmark --iterations 10

# Expected results:
# ├── Average Response Time: 89ms
# ├── 95th Percentile: 156ms
# ├── Throughput: 8.7 req/s
# ├── Error Rate: 2.1%
# └── Cache Hit Rate: 67%
```

### **Load Testing**
```bash
# Install load testing dependencies
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:9090

# Or use built-in load testing
gemini-cli --load-test --users 100 --duration 300
```

### **Stress Testing**
```bash
# Run stress test
gemini-cli --stress-test --max-users 500 --ramp-up 60

# Monitor system resources
htop
iotop
nethogs
```

## 🔧 Troubleshooting

### **Common Issues**

#### **API Key Issues**
```bash
# Check API key configuration
echo $GEMINI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
     "https://generativelanguage.googleapis.com/v1beta/models"

# Verify in .env file
cat .env | grep GEMINI_API_KEY
```

#### **Performance Issues**
```bash
# Check system resources
gemini-cli --stats

# Monitor performance
gemini-cli --monitor

# Check cache status
gemini-cli --cache-status

# Optimize configuration
gemini-cli --optimize
```

#### **Connection Issues**
```bash
# Check network connectivity
ping generativelanguage.googleapis.com

# Test DNS resolution
nslookup generativelanguage.googleapis.com

# Check firewall settings
sudo ufw status
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
gemini-cli --log-level DEBUG --interactive

# Check logs
tail -f logs/gemini-cli.log

# Enable performance profiling
gemini-cli --profile --interactive
```

### **Health Checks**
```bash
# Basic health check
gemini-cli --health

# Detailed health check
gemini-cli --health --verbose

# Component health check
gemini-cli --health --components all

# Circuit breaker status
gemini-cli --health --circuit-breaker
```

## 📈 Monitoring and Observability

### **Prometheus Metrics**
```bash
# Start metrics server
gemini-cli --metrics --port 9090

# View metrics
curl http://localhost:9090/metrics

# Key metrics to monitor:
# - gemini_requests_total
# - gemini_request_duration_seconds
# - gemini_cache_hits_total
# - gemini_active_requests
```

### **Grafana Dashboard**
```bash
# Install Grafana
# On Ubuntu:
sudo apt-get install grafana

# On macOS:
brew install grafana

# Start Grafana
sudo systemctl start grafana-server

# Access dashboard at http://localhost:3000
# Default credentials: admin/admin
```

### **Log Analysis**
```bash
# View real-time logs
tail -f logs/gemini-cli.log

# Search for errors
grep "ERROR" logs/gemini-cli.log

# Search for performance issues
grep "latency\|throughput" logs/gemini-cli.log

# Analyze log patterns
gemini-cli --analyze-logs
```

## 🐳 Docker Deployment

### **Quick Docker Setup**
```bash
# Build image
docker build -t gemini-cli .

# Run container
docker run -it \
  --env-file .env \
  -p 9090:9090 \
  -v $(pwd)/cache:/app/cache \
  gemini-cli

# Run in background
docker run -d \
  --name gemini-cli \
  --env-file .env \
  -p 9090:9090 \
  -v $(pwd)/cache:/app/cache \
  gemini-cli
```

### **Docker Compose**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f gemini-cli

# Scale services
docker-compose up -d --scale gemini-cli=3

# Stop services
docker-compose down
```

## ☸️ Kubernetes Deployment

### **Quick Kubernetes Setup**
```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=gemini-cli

# View logs
kubectl logs -l app=gemini-cli

# Scale deployment
kubectl scale deployment gemini-cli --replicas=5

# Access service
kubectl port-forward svc/gemini-cli-service 8080:80
```

### **Helm Chart**
```bash
# Install Helm chart
helm install gemini-cli ./helm/

# Upgrade deployment
helm upgrade gemini-cli ./helm/

# Uninstall
helm uninstall gemini-cli
```

## 🔒 Security Considerations

### **API Key Security**
```bash
# Use environment variables (recommended)
export GEMINI_API_KEY="your_key_here"

# Use secrets management
# Kubernetes:
kubectl create secret generic gemini-api-secret \
  --from-literal=api-key="your_key_here"

# Docker:
docker run -e GEMINI_API_KEY="your_key_here" gemini-cli
```

### **Network Security**
```bash
# Restrict network access
sudo ufw allow from 192.168.1.0/24 to any port 9090

# Use VPN for remote access
# Configure firewall rules
# Enable TLS encryption
```

## 📚 Examples and Use Cases

### **Basic Text Generation**
```bash
# Simple question
gemini-cli "What is the capital of France?"

# Code generation
gemini-cli "Write a Python function to calculate fibonacci numbers"

# Creative writing
gemini-cli "Write a haiku about technology"
```

### **Advanced Usage**
```bash
# Batch processing
cat > prompts.txt << EOF
Explain quantum computing
Write a Python function for sorting
What are the benefits of renewable energy?
EOF

gemini-cli --batch prompts.txt --output results.json

# Streaming with custom formatting
gemini-cli --stream --format json "Explain machine learning" | jq .

# Performance testing
gemini-cli --benchmark --iterations 100 --output benchmark_results.csv
```

### **Integration Examples**
```bash
# Use in scripts
#!/bin/bash
response=$(gemini-cli "Generate a random password")
echo "Generated password: $response"

# Use in Python
import subprocess
result = subprocess.run(['gemini-cli', 'Explain Python'], 
                       capture_output=True, text=True)
print(result.stdout)

# Use in CI/CD
gemini-cli --benchmark --iterations 50 --threshold 100ms
if [ $? -eq 0 ]; then
    echo "Performance test passed"
else
    echo "Performance test failed"
    exit 1
fi
```

## 🎯 Performance Targets

### **Expected Performance**
```bash
# Target metrics
├── Response Time: < 100ms (TTFB)
├── Throughput: > 5 req/s
├── Error Rate: < 5%
├── Cache Hit Rate: > 50%
├── Memory Usage: < 1GB
└── CPU Usage: < 70%
```

### **Performance Tuning**
```bash
# Optimize for high throughput
export MAX_CONCURRENT_REQUESTS=50
export BATCH_SIZE=20
export CACHE_MAX_SIZE=10000

# Optimize for low latency
export REQUEST_TIMEOUT=15
export KEEP_ALIVE_TIMEOUT=60
export CONNECTION_POOL_SIZE=25

# Balance configuration
export MAX_CONCURRENT_REQUESTS=25
export BATCH_SIZE=10
export CACHE_MAX_SIZE=5000
```

## 🤝 Getting Help

### **Documentation**
- **README.md**: Project overview and quick start
- **DEPLOYMENT.md**: Production deployment guide
- **PERFORMANCE.md**: Performance analysis and optimization
- **API.md**: API reference and examples

### **Support Channels**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **GitHub Wiki**: Additional documentation and tutorials

### **Community**
- **Contributing**: How to contribute to the project
- **Code of Conduct**: Community guidelines
- **Changelog**: Version history and updates

---

## 🚀 Quick Commands Reference

```bash
# Basic usage
gemini-cli --help                    # Show help
gemini-cli --version                 # Show version
gemini-cli --interactive             # Interactive mode
gemini-cli "Your prompt here"        # Direct query

# Performance and monitoring
gemini-cli --status                  # System status
gemini-cli --stats                   # Performance stats
gemini-cli --benchmark               # Run benchmark
gemini-cli --health                  # Health check

# Configuration
gemini-cli --config show             # Show config
gemini-cli --config validate         # Validate config
gemini-cli --config reload           # Reload config

# Advanced features
gemini-cli --stream "Prompt"         # Streaming response
gemini-cli --batch file.txt          # Batch processing
gemini-cli --load-test               # Load testing
gemini-cli --metrics                 # Start metrics server
```

**Happy coding with Gemini CLI! 🚀**

