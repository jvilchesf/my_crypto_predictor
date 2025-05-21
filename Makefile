# Runs the first service
run:
	uv run services/$(service)/src/main.py

build-and-push:
	sh scripts/build-and-push-image.sh $(image_name) $(env)
	
deploy:
	sh scripts/deploy.sh ${service} ${env}	

build-and-deploy:
	sh scripts/build-and-push-image.sh $(service) $(env)
	sh scripts/deploy.sh ${service} ${env}

lint:
	uv run ruff format
	uv run ruff check . --fix