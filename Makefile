# Runs the first service
run:
	uv run services/$(service)/src/main.py

build-and-push:
	sh ./scripts/build-and-push-image.sh ${image} ${env}
	
# Build docker image
build:
	docker build -t ${service}-image -f dockers/${service}.DockerFile .

# push:
# 	Kind load docker-image ${service}-image --name rwml-34fa

deploy: build push
	kubectl delete -f deployments/services/${service}.yaml --ignore-not-found=true
	kubectl apply -f deployments/services/${service}.yaml