# Google Cloud (GCP) — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre o Google Cloud.
**Expert sugerido**: família `gcp` dentro de `cloud_experts` (fase 10). **Total est.**: ~105 lições.
**Convenção**: `treinamento_cloud/gcp/<subsetor>/*.md` → path = [gcp, subsetor].

## fundamentos/ — ~12
o que é o GCP e regiões/zonas; console, `gcloud` CLI e Cloud Shell; projetos e organização; resource hierarchy (org/folder/project); billing e budgets; APIs e serviços; modelo de responsabilidade compartilhada; labels; Google Cloud Architecture Framework; quotas; SDKs; free tier.

## compute/ — ~16
Compute Engine (VMs); machine types; imagens e boot disks; instance groups e autoscaling; preemptible/spot VMs; Cloud Run; Cloud Functions (1ª e 2ª geração); GKE (Kubernetes); GKE Autopilot; App Engine (standard/flexible); Artifact Registry; load balancing; sole-tenant nodes; startup scripts; deploy serverless.

## storage/ — ~11
Cloud Storage (buckets); classes de armazenamento; lifecycle e versionamento; signed URLs; Persistent Disk; Filestore; IAM em buckets; encryption; transfer service; Nearline/Coldline/Archive; static website.

## database/ — ~14
Cloud SQL (MySQL/PostgreSQL/SQL Server); Cloud Spanner; Firestore (modos); Bigtable; Memorystore (Redis); BigQuery (datasets e queries); BigQuery (particionamento/clustering); AlloyDB; backups e réplicas; migração (DMS); escolha do banco; Datastore; consistência; performance.

## networking/ — ~14
VPC (modo auto/custom); subnets; firewall rules; VPC peering; Shared VPC; Cloud Load Balancing (tipos); Cloud CDN; Cloud DNS; Cloud NAT; Cloud Interconnect/VPN; Private Google Access; Network Tags; Cloud Armor; service networking.

## identidade-seguranca/ — ~14
IAM (roles primitivas/predefinidas/custom); service accounts; workload identity; políticas IAM; Secret Manager; Cloud KMS; Identity-Aware Proxy (IAP); Organization Policies; VPC Service Controls; Security Command Center; encryption; least privilege; audit logs; Cloud Identity.

## devops-monitoramento/ — ~13
Deployment Manager; Terraform no GCP; Cloud Build; Cloud Deploy; Artifact Registry; Cloud Monitoring; Cloud Logging; Cloud Trace; Error Reporting; alertas e SLOs; Config Connector; CI/CD; observabilidade.

## data-mensageria/ — ~11
Pub/Sub; Dataflow (Apache Beam); Dataproc (Spark/Hadoop); Cloud Composer (Airflow); Data Fusion; Looker/Looker Studio; Vertex AI (visão geral); Cloud Tasks; Eventarc; padrões event-driven; streaming vs batch.
