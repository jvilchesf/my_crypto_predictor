#!/bin/bash

helm repo add risingwavelabs https://risingwavelabs.github.io/helm-charts/ --force-update

helm repo update

helm upgrade --install --create-namespace --wait risingwave risingwavelabs/risingwave --namespace=risingwave -f manifests/risingwave-values.yaml 

kubectl get secrets -n risingwave risingwave-minio -o json | jq -r '.data."root-password"' | base64 -D
