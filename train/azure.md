# Azure — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre o Microsoft Azure.
**Expert sugerido**: família `azure` dentro de `cloud_experts` (fase 10). **Total est.**: ~110 lições.
**Convenção**: `treinamento_cloud/azure/<subsetor>/*.md` → path = [azure, subsetor].

## fundamentos/ — ~12
o que é o Azure e regiões/zonas; portal, CLI (`az`) e PowerShell; assinaturas e management groups; resource groups; o Azure Resource Manager (ARM); modelo de responsabilidade compartilhada; billing e cost management; tags; Cloud Adoption Framework; Well-Architected; Azure Advisor; políticas e governança (Azure Policy).

## compute/ — ~16
Virtual Machines; VM scale sets; disponibilidade (availability sets/zones); App Service (Web Apps); App Service plans; Azure Functions; Durable Functions; AKS (Kubernetes); Container Instances (ACI); Container Apps; Azure Container Registry; Batch; Service Fabric; Dedicated Host; autoscaling; deployment slots.

## storage/ — ~12
Storage Accounts; Blob Storage; tiers (hot/cool/archive); Azure Files; Queue Storage; Table Storage; Disk Storage (managed disks); redundância (LRS/GRS/ZRS); SAS tokens; lifecycle management; Data Lake Storage; encryption.

## database/ — ~13
Azure SQL Database; SQL Managed Instance; Cosmos DB (APIs e consistência); Cosmos DB (partitioning); Database for PostgreSQL; Database for MySQL; Azure Cache for Redis; Synapse Analytics; elastic pools; backup e geo-replicação; Data Migration Service; escolha do banco; DTU vs vCore.

## networking/ — ~15
Virtual Network (VNet); subnets; Network Security Groups; VNet peering; Load Balancer; Application Gateway (+ WAF); Azure Front Door; Traffic Manager; Azure DNS; VPN Gateway; ExpressRoute; Private Link e endpoints; Azure Firewall; CDN; Bastion.

## identidade-seguranca/ — ~16
Microsoft Entra ID (Azure AD); usuários, grupos e roles; RBAC; managed identities; service principals; conditional access; MFA; Key Vault; Defender for Cloud; Sentinel; encryption; Privileged Identity Management; app registrations; B2C; least privilege.

## devops-monitoramento/ — ~15
ARM templates; Bicep; Azure DevOps (Boards/Repos/Pipelines); GitHub Actions no Azure; Azure Monitor; Application Insights; Log Analytics (KQL); alertas; Automation; Blueprints; deployment strategies; Terraform no Azure; dashboards; Resource Health; auditoria.

## mensageria-integracao/ — ~11
Service Bus (filas/tópicos); Event Grid; Event Hubs; Logic Apps; API Management; SignalR; Notification Hubs; Storage Queues vs Service Bus; padrões event-driven; Data Factory; Stream Analytics.
