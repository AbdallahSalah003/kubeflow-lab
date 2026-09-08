# Kubeflow Lab

A Kubeflow Pipelines project for the Telco Customer Churn dataset.

## Project Structure

```text
├── pipeline.py
├── telco_churn_pipeline.yaml
├── pyproject.toml
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── collection.py
    │   ├── validation.py
    │   └── cleaning.py
    ├── features/
    │   ├── __init__.py
    │   └── features.py
    ├── train/
    │   ├── __init__.py
    │   └── train.py
    └── eval/
        ├── __init__.py
        └── eval.py
```

## Kubeflow Pipeline Run
<img source="kubeflow-pipeline-run1.png" alt="">
<img source="kubeflow-pipeline-run2.png" alt="">



### Prerequisites

EC2 Instance Type: m7i-flex.large
OS: Ubuntu 22.04 or 24.04 LTS


### 1. Install dependencies

```bash
sudo apt-get update
sudo apt-get install -y curl git ca-certificates gnupg lsb-release
```

### 2. Install Docker

```bash
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo usermod -aG docker "$USER"
```

Log out and back in, then verify:

```bash
docker --version
```

### 3. Install kubectl

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

kubectl version --client
```

### 4. Install Kind

```bash
curl -Lo kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
chmod +x kind
sudo mv kind /usr/local/bin/kind

kind version
```

### 5. Create the Kind cluster

```bash
kind create cluster --name kubeflow-lab --wait 5m
```

Verify:

```bash
kubectl cluster-info --context kind-kubeflow-lab
kubectl get nodes
```

### 6. Install Kubeflow

Refer to https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/ for the latest version
```bash
export PIPELINE_VERSION=2.17.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```


Wait for the workloads to become ready

```bash
kubectl get pods -A
```

### 7. Access Kubeflow Pipelines

Since the Kind cluster is running on EC2, `127.0.0.1` refers to the EC2 instance. To access the UI from your local machine 
create an SSH tunnel:

```bash
ssh -L 8080:localhost:8080 <user>@<EC2_PUBLIC_IP>
```

Then open:

```text
http://127.0.0.1:8080
```

## Cleanup

```bash
kind delete cluster --name kubeflow-lab
```
