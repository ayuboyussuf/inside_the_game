// Hunt Autopsy on Azure Container Apps + Azure OpenAI (managed identity, no keys).
// Deploy: ./infra/deploy.sh   (or: az deployment group create -g <rg> -f infra/main.bicep -p image=<acr>/hunt-autopsy:latest)
targetScope = 'resourceGroup'

@description('Base name for resources')
param name string = 'huntautopsy'
param location string = resourceGroup().location
@description('Container image, e.g. myacr.azurecr.io/hunt-autopsy:latest')
param image string
@description('Azure OpenAI model deployment')
param openAiModel string = 'gpt-4o-mini'
param openAiModelVersion string = '2024-07-18'
param openAiLocation string = location

var suffix = uniqueString(resourceGroup().id)

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${name}-logs-${suffix}'
  location: location
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${name}${suffix}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${name}-id'
  location: location
}

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${name}-aoai-${suffix}'
  location: openAiLocation
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: { customSubDomainName: '${name}-aoai-${suffix}', publicNetworkAccess: 'Enabled', disableLocalAuth: true }
}

resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: openAiModel
  sku: { name: 'GlobalStandard', capacity: 30 }
  properties: { model: { format: 'OpenAI', name: openAiModel, version: openAiModelVersion } }
}

// Cognitive Services OpenAI User
resource aoaiRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openai.id, identity.id, 'openai-user')
  scope: openai
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  }
}

// AcrPull
resource acrRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, 'acrpull')
  scope: acr
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${name}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  dependsOn: [ acrRole, aoaiRole ]
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: [ { server: acr.properties.loginServer, identity: identity.id } ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: deployment.name }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
          ]
          probes: [ { type: 'Readiness', httpGet: { path: '/api/health', port: 8000 }, initialDelaySeconds: 5, periodSeconds: 10 } ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 5, rules: [ { name: 'http', http: { metadata: { concurrentRequests: '40' } } } ] }
    }
  }
}

output url string = 'https://${app.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
