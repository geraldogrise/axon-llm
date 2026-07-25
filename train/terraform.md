# Terraform — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Terraform (IaC).
**Expert sugerido**: família `terraform` dentro de `devops_experts` (fase 11). **Total est.**: ~70 lições.
**Convenção**: `treinamento_devops/terraform/<subsetor>/*.md` → path = [terraform, subsetor].

## fundamentos/ — ~14
o que é Infrastructure as Code; o que é Terraform e como funciona; instalação e `terraform` CLI; HCL (sintaxe); providers; o primeiro resource; `init`/`plan`/`apply`/`destroy`; variables (input); tipos de variáveis; outputs; locals; data sources; interpolação e referências; o fluxo de trabalho básico.

## estado/ — ~10
o que é o state; o arquivo `terraform.tfstate`; remote state (S3/GCS/azurerm); state locking (DynamoDB); backends; `terraform state` (mv/rm/list); import de recursos existentes; drift e refresh; state em equipe; segurança do state (secrets).

## modulos/ — ~10
o que são módulos; criar um módulo; usar módulos (source); input e output de módulos; módulos do registry; versionamento de módulos; composição de módulos; módulos aninhados; boas práticas de estrutura; reutilização.

## linguagem-hcl/ — ~16
meta-argumentos: `count`; `for_each`; `depends_on`; `lifecycle` (create_before_destroy/prevent_destroy); expressões condicionais; `for` expressions; funções built-in (string/collection/numeric); dynamic blocks; `templatefile`; tipos complexos (list/map/object); validação de variáveis; sensitive values; `terraform_data`/null_resource; provisioners (local-exec/remote-exec); moved blocks.

## workflow-producao/ — ~12
workspaces; múltiplos ambientes (dev/stage/prod); `terraform fmt` e `validate`; `-target` e `-replace`; variáveis de ambiente e `.tfvars`; CI/CD com Terraform; Terraform Cloud/Enterprise; políticas (Sentinel/OPA); testes (terratest); debugging (`TF_LOG`); versionamento de providers; migração de versões.

## boas-praticas/ — ~8
estrutura de projeto; convenções de nomenclatura; DRY e módulos; gestão de secrets; separação de estado; code review de IaC; documentação; antipadrões comuns.
