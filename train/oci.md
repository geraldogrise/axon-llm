# Oracle Cloud (OCI) — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre a Oracle Cloud Infrastructure.
**Expert sugerido**: família `oci` dentro de `cloud_experts` (fase 10). **Total est.**: ~100 lições.
**Convenção**: `treinamento_cloud/oci/<subsetor>/*.md` → path = [oci, subsetor].

## fundamentos/ — ~12
o que é a OCI e regiões/domínios de disponibilidade; console, CLI (`oci`) e Cloud Shell; tenancy e compartments; hierarquia de recursos; billing e cost management; modelo de responsabilidade compartilhada; tags; fault domains; SDKs; free tier (Always Free); quotas e limites; a arquitetura da OCI.

## compute/ — ~15
Compute Instances (VM/bare metal); shapes (flexíveis); imagens e custom images; instance pools e autoscaling; preemptible instances; Container Instances; OKE (Kubernetes Engine); OCI Functions (serverless); Container Registry (OCIR); block volume attach; cloud-init; dedicated hosts; instance configurations; deploy.

## storage/ — ~11
Object Storage (buckets e tiers); Archive Storage; Block Volume; volume performance tiers; File Storage; backups e clones; pre-authenticated requests; replicação; lifecycle policies; encryption; Data Transfer.

## database/ — ~15
Autonomous Database (ATP/ADW); Autonomous scaling; Base Database Service; Exadata Database Service; MySQL HeatWave; PostgreSQL; NoSQL Database; Database backups; Data Guard (DR); RAC; migração (ZDM); APEX; SQL Developer Web; escolha do banco; performance.

## networking/ — ~14
Virtual Cloud Network (VCN); subnets (públicas/privadas); Internet Gateway e NAT Gateway; Service Gateway; route tables; Security Lists vs Network Security Groups; Load Balancer; DNS; FastConnect; Site-to-Site VPN; Local/Remote peering; Web Application Firewall; DRG (Dynamic Routing Gateway); private endpoints.

## identidade-seguranca/ — ~14
IAM (usuários, grupos, compartments); políticas IAM (sintaxe OCI); dynamic groups; instance principals; federação (Identity Domains); Vault e KMS; Secrets; Cloud Guard; Security Zones; MFA; encryption; least privilege; audit; Bastion service.

## devops-monitoramento/ — ~13
Resource Manager (Terraform gerenciado); OCI DevOps service; deployment pipelines; Container Registry; Monitoring (métricas e alarmes); Logging; Logging Analytics; Notifications; Events service; Health Checks; Application Performance Monitoring; CI/CD; observabilidade.
