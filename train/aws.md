# AWS — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre AWS.
**Expert sugerido**: família `aws` dentro de `cloud_experts` (fase 10). **Total est.**: ~120 lições.
**Convenção**: `treinamento_cloud/aws/<subsetor>/*.md` → path = [aws, subsetor].

## fundamentos/ — ~12
o que é a AWS e regiões/AZs; console, CLI e SDK; modelo de responsabilidade compartilhada; contas e Organizations; billing e cost management; o AWS Well-Architected Framework; tags e resource groups; limites e quotas; a AWS CLI (configuração e perfis); CloudShell; suporte e SLAs; arquitetura de alta disponibilidade.

## compute/ — ~18
EC2 (instâncias e tipos); AMIs; key pairs e user data; Auto Scaling Groups; Elastic Load Balancing (ALB/NLB); EC2 pricing (on-demand/spot/reserved); EBS (volumes); Lambda (funções); Lambda triggers e layers; API Gateway + Lambda (serverless); ECS (containers); Fargate; EKS (Kubernetes); ECR; Elastic Beanstalk; Batch; Lightsail; placement groups.

## storage/ — ~14
S3 (buckets e objetos); classes de armazenamento S3; versionamento e lifecycle; políticas de bucket e ACLs; S3 static website; presigned URLs; EBS tipos e snapshots; EFS; FSx; Storage Gateway; Glacier e arquivamento; S3 replication; transfer acceleration; encryption em repouso.

## database/ — ~14
RDS (engines e Multi-AZ); read replicas; Aurora; DynamoDB (tabelas e chaves); DynamoDB (índices GSI/LSI); DynamoDB streams; ElastiCache (Redis/Memcached); Redshift; DocumentDB; Neptune; backups e snapshots; RDS Proxy; migração (DMS); escolha do banco certo.

## networking/ — ~16
VPC (subnets públicas/privadas); Internet Gateway e NAT; route tables; Security Groups vs NACLs; VPC peering; Transit Gateway; Route 53 (DNS e políticas); CloudFront (CDN); Direct Connect; VPN; Elastic IPs; ENI; PrivateLink e endpoints; Global Accelerator; load balancing deep; DNS failover.

## seguranca-iam/ — ~16
IAM (usuários, grupos, roles); políticas IAM (JSON); roles e assume role; STS; MFA; KMS (chaves); Secrets Manager; Parameter Store; Cognito (auth de usuários); WAF; Shield; GuardDuty; Security Hub; least privilege; cross-account access; encryption em trânsito.

## devops-monitoramento/ — ~16
CloudFormation (IaC); CDK; CloudWatch (métricas e alarmes); CloudWatch Logs; CloudTrail (auditoria); Systems Manager; CodeCommit/CodeBuild/CodeDeploy/CodePipeline; X-Ray (tracing); EventBridge; Config; tagging strategy; deployment strategies (blue/green); rollback; Trusted Advisor; observabilidade.

## mensageria-integracao/ — ~14
SQS (filas); SNS (pub/sub); EventBridge (eventos); Kinesis (streaming); Kinesis Firehose; SES (e-mail); Step Functions (orquestração); MQ; API Gateway (REST/HTTP/WebSocket); AppSync (GraphQL); MSK (Kafka); dead-letter queues; fanout; padrões event-driven.
