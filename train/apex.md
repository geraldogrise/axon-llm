# Apex (Salesforce) — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Apex (Salesforce).
**Expert sugerido**: família em `enterprise_experts`. **Total est.**: ~55 lições.
**Convenção**: `treinamento_apex/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~22
o que é a plataforma Salesforce e Apex; sintaxe (Java-like); variáveis e tipos; sObjects; primitivos; coleções (List/Set/Map); operadores; controle de fluxo; loops; classes e métodos; construtores; herança; interfaces; propriedades; exceptions; estático vs instância; enums; comentários; governor limits (fundamental).

## sfdc-dados/ — ~21
SOQL (queries); SOQL relationships; SOSL (search); DML (insert/update/delete); DML e bulkification; sObject e campos; upsert; database methods; transações e rollback; triggers; contextos de trigger; trigger patterns; batch Apex; queueable Apex; scheduled Apex; future methods; assíncrono; platform events; callouts (REST/SOAP); JSON.

## avancado-ecossistema/ — ~12
testes (test classes e cobertura); test data; mocking; Lightning Web Components (visão geral); Aura; controllers (@AuraEnabled); segurança (CRUD/FLS/sharing); with/without sharing; governor limits (detalhado); design patterns; deployment (metadata); SFDX; boas práticas.
