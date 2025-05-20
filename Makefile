# Runs the first service
run:
	uv run services/$(service)/src/main.py

build-and-push:
	sh deployments/scripts/build-and-push-image.sh $(image_name) $(env)
	
deploy-historical:
	sh deployments/scripts/deploy.sh ${service} ${env}	
