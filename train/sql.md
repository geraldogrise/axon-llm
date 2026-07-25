# SQL — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre SQL (padrão + dialetos).
**Expert sugerido**: `sql_experts` ou família em `db_experts`. **Total est.**: ~100 lições.
**Convenção**: `treinamento_sql/<família>/<subsetor>/*.md` → path = [família, subsetor].

## fundamentos/ — ~14
o que é SQL e bancos relacionais; SELECT básico; WHERE e filtros; operadores de comparação e lógicos; ORDER BY; LIMIT/OFFSET; DISTINCT; aliases; NULL e IS NULL; BETWEEN/IN/LIKE; comentários; tipos de dados; expressões e aritmética.

## ddl/ — ~12
CREATE TABLE; tipos de coluna; constraints (PRIMARY KEY/NOT NULL/UNIQUE); FOREIGN KEY; DEFAULT e CHECK; ALTER TABLE; DROP e TRUNCATE; índices (CREATE INDEX); views; sequences/auto-increment; schemas; tabelas temporárias.

## dml-consultas/ — ~20
INSERT; UPDATE; DELETE; UPSERT (MERGE/ON CONFLICT); INNER JOIN; LEFT/RIGHT/FULL JOIN; CROSS JOIN e self join; GROUP BY; HAVING; funções de agregação (COUNT/SUM/AVG/MIN/MAX); subqueries; subqueries correlacionadas; CTEs (WITH); CTEs recursivas; UNION/INTERSECT/EXCEPT; CASE; COALESCE/NULLIF; EXISTS; ANY/ALL.

## avancado/ — ~24
window functions (OVER/PARTITION BY); ROW_NUMBER/RANK/DENSE_RANK; LAG/LEAD; running totals; funções de string; funções de data/hora; funções numéricas; pivot/unpivot; JSON em SQL; full-text search; transações (COMMIT/ROLLBACK); níveis de isolamento; locks; stored procedures; functions; triggers; cursors; GRANT/REVOKE (permissões); explain plan.

## performance-modelagem/ — ~18
índices (como funcionam); tipos de índice (B-tree/hash/GIN); quando indexar; EXPLAIN e query plans; otimização de consultas; normalização (1FN/2FN/3FN); desnormalização; modelagem de dados; relacionamentos (1:1/1:N/N:N); chaves e integridade referencial; particionamento; sharding (visão geral); dialetos (PostgreSQL vs MySQL vs SQL Server vs Oracle); ANSI SQL vs proprietário; boas práticas; anti-padrões.
