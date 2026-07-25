# ABAP — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre ABAP (SAP).
**Expert sugerido**: família em `enterprise_experts`. **Total est.**: ~55 lições.
**Convenção**: `treinamento_abap/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~24
o que é ABAP e o ecossistema SAP; a estrutura de um programa; tipos de dados; declaração (DATA); variáveis e constantes; operadores; strings; controle de fluxo (IF/CASE); loops (DO/WHILE/LOOP); subroutines (FORM); function modules; parâmetros; internal tables; work areas; estruturas; field symbols; MOVE-CORRESPONDING; WRITE; mensagens; comentários.

## dados-sap/ — ~19
Data Dictionary (DDIC); tabelas transparentes; Open SQL (SELECT); JOINs; internal tables (operações); READ TABLE; SORT/DELETE/MODIFY; ALV (relatórios); selection screens; eventos de programa; classic reports; module pool; BAPIs; IDocs; enhancement; BADIs; user exits; autorização.

## abap-oo-ecossistema/ — ~12
ABAP Objects (OOP); classes e métodos; herança; interfaces; exceptions (classes); ABAP moderno (7.4+); inline declarations; CDS Views; AMDP; RAP (RESTful ABAP); SAP HANA; SAP Fiori (visão geral); testes (ABAP Unit); boas práticas; migração para S/4HANA.
