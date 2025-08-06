# Model Operator

A Kubernetes operator for deploying and managing machine learning models in production environments. This operator provides a declarative way to deploy ML models with support for multiple inference runtimes including Triton, TorchServe, and FastAPI-PyTorch.

## 🚀 Overview

The Model Operator is built using the [Kubebuilder](https://book.kubebuilder.io/) framework and provides:

- **Custom Resource Definition (CRD)**: `ModelDeployment` for declarative model deployment
- **Multiple Runtime Support**: Triton, TorchServe, and FastAPI-PyTorch
- **Model Validation**: Optional validation scripts before deployment
- **Autoscaling**: Built-in horizontal pod autoscaling support
- **Monitoring**: Prometheus metrics integration
- **Resource Management**: Configurable CPU and memory limits

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Go** 1.24.0 or later
- **Docker** or **Podman** for container operations
- **kubectl** configured to communicate with your cluster
- **make** for running build commands
- **Kind** (optional, for local testing)

## 🏗️ Architecture

### Custom Resource: ModelDeployment

The operator defines a `ModelDeployment` custom resource with the following specification:

```yaml
apiVersion: mlops.sathvik.dev/v1
kind: ModelDeployment
metadata:
  name: example-model
spec:
  modelURI: "https://example.com/model.pt" # Model artifact location
  runtime: "fastapi-pytorch" # Inference runtime
  resources: # Resource requirements
    limits:
      cpu: "500m"
      memory: "512Mi"
    requests:
      cpu: "250m"
      memory: "256Mi"
  validateScript: "validate.py" # Optional validation script
  autoscale: true # Enable autoscaling
  version: "v1" # Model version
  runtimeImage: "custom-image:tag" # Optional runtime image override
```

### Supported Runtimes

1. **triton**: NVIDIA Triton Inference Server
2. **torchserve**: PyTorch TorchServe
3. **fastapi-pytorch**: Custom FastAPI with PyTorch

## 🛠️ Development Setup

### 1. Clone and Setup

```bash
git clone <repository-url>
cd model-operator
```

### 2. Install Dependencies

The project uses `make` targets to manage dependencies. Run:

```bash
# Install all required tools (kustomize, controller-gen, etc.)
make controller-gen
make kustomize
make setup-envtest
make golangci-lint
```

### 3. Generate Code

```bash
# Generate CRDs and RBAC manifests
make manifests

# Generate DeepCopy methods
make generate
```

### 4. Build and Test

```bash
# Build the operator binary
make build

# Run tests
make test

# Run linting
make lint

# Run e2e tests (requires Kind cluster)
make test-e2e
```

## 🚀 Deployment

### Local Development

```bash
# Run the operator locally
make run
```

### Deploy to Kubernetes Cluster

```bash
# Install CRDs
make install

# Deploy the operator
make deploy

# Or build and deploy with custom image
make docker-build IMG=your-registry/model-operator:latest
make deploy IMG=your-registry/model-operator:latest
```

### Build Multi-Platform Image

```bash
# Build for multiple architectures
make docker-buildx IMG=your-registry/model-operator:latest
```

## 📦 Usage Examples

### 1. Deploy a Simple Model

Create a `ModelDeployment` resource:

```yaml
apiVersion: mlops.sathvik.dev/v1
kind: ModelDeployment
metadata:
  name: dummy-model
spec:
  modelURI: https://dummy.com/dummy_model.pt
  runtime: fastapi-pytorch
  resources:
    limits:
      cpu: "500m"
      memory: "512Mi"
    requests:
      cpu: "250m"
      memory: "256Mi"
  autoscale: false
  version: v1
```

Apply it to your cluster:

```bash
kubectl apply -f example-cr.yaml
```

### 2. Deploy with Custom Runtime Image

```yaml
apiVersion: mlops.sathvik.dev/v1
kind: ModelDeployment
metadata:
  name: custom-model
spec:
  modelURI: https://example.com/model.pt
  runtime: fastapi-pytorch
  runtimeImage: "localhost/fastapi-pytorch:dev"
  resources:
    limits:
      cpu: "1"
      memory: "1Gi"
    requests:
      cpu: "500m"
      memory: "512Mi"
  validateScript: "validate.py"
  autoscale: true
  version: v1
```

### 3. Monitor Deployment Status

```bash
# Check ModelDeployment status
kubectl get modeldeployments

# Get detailed information
kubectl describe modeldeployment dummy-model

# Check operator logs
kubectl logs -n model-operator-system deployment/model-operator-controller-manager
```

## 🔧 Configuration

### Runtime Images

The operator uses predefined images for each runtime:

- **fastapi-pytorch**: `localhost/fastapi-pytorch:dev`
- **triton**: `nvcr.io/nvidia/tritonserver:latest`
- **torchserve**: `pytorch/torchserve:latest`

You can override these using the `runtimeImage` field.

### Resource Requirements

Configure CPU and memory limits/requests based on your model's requirements:

```yaml
resources:
  limits:
    cpu: "2"
    memory: "4Gi"
  requests:
    cpu: "1"
    memory: "2Gi"
```

### Validation Scripts

Optional validation scripts can be specified to validate models before deployment:

```yaml
validateScript: "validate.py"
```

## 📊 Monitoring

The operator exposes Prometheus metrics:

- `model_deployments_created_total`: Total number of ModelDeployment resources created
- `model_validations_total`: Number of model validation jobs run (with status labels)

### Accessing Metrics

```bash
# Port forward to access metrics
kubectl port-forward -n model-operator-system deployment/model-operator-controller-manager 8080:8080

# Access metrics endpoint
curl http://localhost:8080/metrics
```

## 🧪 Testing

### Unit Tests

```bash
# Run unit tests
make test
```

### E2E Tests

```bash
# Setup Kind cluster and run e2e tests
make test-e2e

# Cleanup test cluster
make cleanup-test-e2e
```

### Manual Testing

```bash
# Deploy example model
kubectl apply -f example-cr.yaml

# Check deployment status
kubectl get modeldeployments
kubectl get pods
kubectl get services

# Test inference (if using FastAPI runtime)
kubectl port-forward service/dummy-model-service 8000:8000
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0], [2.0]]}'
```

## 🗂️ Project Structure

```
model-operator/
├── api/v1/                    # API definitions and CRDs
├── cmd/main.go               # Operator entry point
├── internal/controller/      # Controller logic
├── config/                   # Kustomize configurations
│   ├── crd/                 # CRD manifests
│   ├── rbac/                # RBAC configurations
│   └── manager/             # Manager deployment
├── inference/               # Example inference service
├── test/                    # Test files
├── charts/                  # Helm charts
└── Makefile                 # Build and deployment commands
```

## 🔍 Troubleshooting

### Common Issues

1. **CRD not found**: Ensure CRDs are installed with `make install`
2. **Image pull errors**: Check if runtime images are accessible
3. **Resource constraints**: Verify cluster has sufficient resources
4. **Validation failures**: Check validation script logs

### Debug Commands

```bash
# Check operator logs
kubectl logs -n model-operator-system deployment/model-operator-controller-manager

# Check CRD status
kubectl get crd modeldeployments.mlops.sathvik.dev

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Describe resources
kubectl describe modeldeployment <name>
kubectl describe deployment <name>
kubectl describe service <name>
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run linting: `make lint`
6. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🔗 Related Projects

- [Kubebuilder](https://book.kubebuilder.io/) - Framework for building Kubernetes operators
- [NVIDIA Triton](https://github.com/triton-inference-server/server) - Inference server
- [TorchServe](https://github.com/pytorch/serve) - PyTorch model serving
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
