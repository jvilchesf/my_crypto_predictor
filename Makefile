# Runs the first service
run:
	uv run services/${service}/src/main.py

# Build docker image
build:
	docker build -t get-trades-image -f dockers/trades.DockerFile .

push:
	Kind load docker-image get-trades-image --name rwml-34fa

deploy: build push
	kubectl delete -f deployments/services/get_trades.yaml --ignore-not-found=true
	kubectl apply -f deployments/services/get_trades.yaml
