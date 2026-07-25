# Kubernetes (K8s) — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Kubernetes.
**Expert sugerido**: família `kubernetes` dentro de `devops_experts` (fase 11). **Total est.**: ~95 lições.
**Convenção**: `treinamento_devops/kubernetes/<subsetor>/*.md` → path = [kubernetes, subsetor].

## fundamentos/ — ~14
o que é o Kubernetes e por que existe; arquitetura (control plane e nodes); componentes (api-server/etcd/scheduler/controller-manager/kubelet/kube-proxy); clusters (minikube/kind); `kubectl` (comandos essenciais); manifestos YAML; namespaces; labels e selectors; annotations; o objeto Pod; multi-container pods; init containers; kubeconfig e contexts; a API declarativa.

## workloads/ — ~16
Pods (ciclo de vida); ReplicaSets; Deployments; rolling updates e rollback; estratégias de deploy; StatefulSets; DaemonSets; Jobs; CronJobs; escalonamento manual; ReplicationController (legado); pod disruption budgets; deployment revisions; recriar vs rolling; garbage collection.

## rede-servicos/ — ~14
Services (ClusterIP/NodePort/LoadBalancer); Service discovery e DNS; Endpoints; Ingress; Ingress controllers (nginx); regras de Ingress e TLS; Network Policies; port-forward; headless services; ExternalName; CNI (visão geral); comunicação pod-a-pod; Gateway API; service mesh (visão geral).

## config-storage/ — ~14
ConfigMaps; Secrets; variáveis de ambiente a partir de ConfigMap/Secret; volumes; emptyDir e hostPath; PersistentVolumes (PV); PersistentVolumeClaims (PVC); StorageClasses; provisionamento dinâmico; volumes em StatefulSets; secrets encryption; mount de arquivos; downward API.

## escalonamento-scheduling/ — ~12
resource requests e limits; Horizontal Pod Autoscaler (HPA); Vertical Pod Autoscaler; Cluster Autoscaler; node selectors; affinity e anti-affinity; taints e tolerations; QoS classes; probes (liveness/readiness/startup); priority e preemption; topology spread; scheduling manual.

## seguranca-rbac/ — ~13
RBAC (Roles e RoleBindings); ClusterRoles; ServiceAccounts; autenticação e autorização; Pod Security Standards; Security Contexts; secrets management; admission controllers; Network Policies (segurança); imagem e supply chain; least privilege; auditoria; OPA/Gatekeeper.

## ecossistema-operacoes/ — ~12
Helm (charts e releases); Helm templates e values; Kustomize; operadores e CRDs; observabilidade (metrics-server/Prometheus); logging; kubectl debugging; troubleshooting de pods; managed K8s (EKS/AKS/GKE/OKE); GitOps (ArgoCD/Flux); upgrades de cluster; backup (Velero).
