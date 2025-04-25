# 1. Delete the cluster (if it exists, otherwise it will fail)
echo "Deleting the cluster..."
kind delete cluster --name rwml-34fa

# 2. Delete the docker network (if it exists, otherwise it will fail)
echo "Deleting the docker network..."
docker network rm rwml-34fa-network

# 3. Create the docker network
echo "Creating the docker network..."
docker network create --subnet 172.100.0.0/16 rwml-34fa-network

KIND_EXPERIMENTAL_DOCKER_NETWORK=rwml-34fa-network kind create cluster --config manifests/ports_cluster_config.yaml



