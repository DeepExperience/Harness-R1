docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller webshop-env_train alfworld-env_train

python -m src.assigner -r -c configs/assignments/alfworld_train.yaml
python -m src.assigner -r -c configs/assignments/webshop_train.yaml

python -m src.assigner -r -c configs/assignments/alfworld_train_harness.yaml
