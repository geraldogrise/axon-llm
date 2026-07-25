# COBOL — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre COBOL (mainframe/legado).
**Expert sugerido**: família em `legacy_experts`. **Total est.**: ~55 lições.
**Convenção**: `treinamento_cobol/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~25
o que é COBOL e onde roda (mainframe); estrutura de um programa; as 4 divisions (IDENTIFICATION/ENVIRONMENT/DATA/PROCEDURE); colunas e formato fixo; PICTURE clauses; níveis de dados (01/05/...); WORKING-STORAGE; variáveis numéricas e alfanuméricas; MOVE; COMPUTE e aritmética; IF/ELSE; PERFORM (loops); PERFORM THRU; paragraphs e sections; DISPLAY/ACCEPT; comentários.

## dados-arquivos/ — ~18
FILE SECTION e FD; arquivos sequenciais; arquivos indexados (VSAM); OPEN/READ/WRITE/CLOSE; record layouts; REDEFINES; OCCURS (tabelas/arrays); indexação de tabelas; SEARCH; STRING/UNSTRING; INSPECT; edição de campos numéricos; COPY (copybooks); tratamento de EOF; matching records.

## avancado-legado/ — ~12
CALL (subprogramas); parâmetros (LINKAGE SECTION); COBOL e CICS (transações); COBOL e DB2 (SQL embutido); JCL (visão geral); COBOL estruturado vs GO TO; tratamento de erros; COBOL moderno (GnuCOBOL); modernização/migração; boas práticas; debugging; performance.
