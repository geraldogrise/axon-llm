# PowerShell — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre PowerShell.
**Expert sugerido**: família em `devops_experts` ou `shell_experts`. **Total est.**: ~85 lições.
**Convenção**: `treinamento_powershell/<família>/<subsetor>/*.md` → path = [família, subsetor].

## fundamentos/ — ~16
o que é o PowerShell (Windows/Core); cmdlets e verbo-substantivo; o pipeline de objetos; Get-Help e Get-Command; variáveis (`$var`); tipos de dados; operadores; aspas e escaping; arrays; hashtables; Get-Member; Select-Object; Where-Object; ForEach-Object; formatação de saída; providers e PSDrives.

## scripting/ — ~24
scripts `.ps1`; execution policy; funções; parâmetros e `param()`; parâmetros avançados (validação); condicionais (`if`/`switch`); loops (`for`/`foreach`/`while`); tratamento de erros (`try`/`catch`); `$?` e `$Error`; `-ErrorAction`; retorno de valores; comparação de operadores; expressões regulares; strings (formatação/here-strings); splatting; scopes; módulos; comment-based help; debugging.

## objetos-sistema/ — ~24
trabalhar com objetos; propriedades e métodos; criar objetos customizados (PSCustomObject); classes (PS 5+); .NET no PowerShell; WMI/CIM; registro do Windows; serviços; processos; arquivos e pastas; Get-Content/Set-Content; CSV (Import/Export-Csv); JSON (ConvertTo/From-Json); XML; datas e horas; comparação e filtro; medição (Measure-Object); Group-Object; Sort-Object.

## administracao-remoto/ — ~21
administração do Windows; Active Directory (módulo); gerenciamento de usuários; remoting (Invoke-Command); PSSession; credenciais seguras (Get-Credential); SecureString; agendamento de tarefas; event logs; Azure PowerShell; Exchange/M365 (visão geral); REST APIs (Invoke-RestMethod); download (Invoke-WebRequest); DSC (Desired State Config); PSScriptAnalyzer; assinatura de scripts; automação; boas práticas.
