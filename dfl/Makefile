DOCKER_IMAGE_NAME := dlf
DOCKER_CONTAINER_NAME := $(DOCKER_IMAGE_NAME)--vm

FILEPATH :=

build:
	cd docker && sudo docker-compose build --no-cache

up:
	cd docker && sudo docker-compose up

exec:
	cd docker && sudo docker-compose exec dev bash

submit:
	$(call __require_FILEPATH)
	kaggle competitions submit -c h-and-m-personalized-fashion-recommendations -f $(FILEPATH) -m "Message"


define __require_FILEPATH
    @bash -c "if [ '${FILEPATH}' = '' ]; then echo 'FILEPATH is not defined; you must specify FILEPATH like $$ make PARAM=xxx task'; exit 1; fi"
endef
