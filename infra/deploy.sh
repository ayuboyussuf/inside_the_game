#!/usr/bin/env bash
# Build in ACR and deploy to Azure Container Apps.
# Usage: RG=hunt-autopsy-rg LOCATION=swedencentral ./infra/deploy.sh
set -euo pipefail
RG=${RG:-hunt-autopsy-rg}
LOCATION=${LOCATION:-swedencentral}
NAME=${NAME:-huntautopsy}

az group create -n "$RG" -l "$LOCATION" -o none
# 1st pass creates the registry (placeholder image), then build into it, then point the app at the real image.
ACR=$(az deployment group create -g "$RG" -f infra/main.bicep -p name="$NAME" image=mcr.microsoft.com/k8se/quickstart:latest \
      --query properties.outputs.acrLoginServer.value -o tsv)
az acr build -r "${ACR%%.*}" -t hunt-autopsy:latest .
URL=$(az deployment group create -g "$RG" -f infra/main.bicep -p name="$NAME" image="$ACR/hunt-autopsy:latest" \
      --query properties.outputs.url.value -o tsv)
echo "Deployed: $URL"
