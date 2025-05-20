#!/bin/bash

# Build a docker for the given image and push it into the Docker registry.

image_name=$1
env=$2  # dev or prod

echo "Image name: $image_name"
echo "Environment: $env"

if [ -z "$image_name" ]; then
    echo "Usage: $0 <image_name> <env>"
    exit 1
fi

if [ -z "$env" ]; then
    echo "Usage: $0 <image_name> <env>"
    exit 1
fi

# Change to the project root directory
#cd ..

# Check that env is either "dev" or "prod"
if [ "$env" = "dev" ]; then  
    echo "deploying image for dev"
    kubectl delete -f deployments/services/${image_name}.yaml --ignore-not-found=true
	kubectl apply -f deployments/services/${image_name}.yaml

else
    echo "deploying image for ${image_name} for prod"
    BUILD_DATE=$(date +%s)
    docker build -t ${image_name}-image:${BUILD_DATE} -f dockers/${image_name}.DockerFile .
    docker tag ${image_name}-image:${BUILD_DATE} your-dockerhub-username/${image_name}:${BUILD_DATE}
    docker push your-dockerhub-username/${image_name}:${BUILD_DATE}
fi