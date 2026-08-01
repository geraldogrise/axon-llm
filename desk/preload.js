// Ponte estreita para a página. A porta e o token só saem daqui já embutidos nas
// chamadas -- o JavaScript da página nunca vê o token.

const { contextBridge, ipcRenderer } = require("electron");

let conexao = null;

async function garantir() {
  if (!conexao) conexao = await ipcRenderer.invoke("conexao");
  if (!conexao) throw new Error("backend indisponível");
  return conexao;
}

async function pedir(rota, corpo) {
  const c = await garantir();
  const r = await fetch(`http://127.0.0.1:${c.porta}${rota}`, {
    method: corpo ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${c.token}`,
      "Content-Type": "application/json",
    },
    body: corpo ? JSON.stringify(corpo) : undefined,
  });
  return r.json();
}

contextBridge.exposeInMainWorld("axon", {
  painel: () => ipcRenderer.invoke("painel"),
  saude: () => pedir("/health"),
  perguntar: (texto, modelo) => pedir("/chat", { texto, modelo }),
  executar: (comando, cwd) => pedir("/exec", { comando, cwd }),

  // O terminal manda tudo por aqui: o backend decide se é comando ou pergunta.
  linha: (texto, cwd, modelo) => pedir("/code", { texto, cwd, modelo }),

  cancelar: (run_id) => pedir("/cancel", { run_id }),

  // O EventSource não manda cabeçalho, então o token vai na query -- é loopback e o
  // token é efêmero, criado a cada subida.
  ouvir: async (aoEvento) => {
    const c = await garantir();
    const es = new EventSource(
      `http://127.0.0.1:${c.porta}/events?token=${c.token}`
    );
    es.onmessage = (e) => aoEvento(JSON.parse(e.data));
    ["boot", "usuario", "rota", "delta", "fim", "erro", "cancelado",
     "comando", "saida", "comando_fim"].forEach((t) =>
      es.addEventListener(t, (e) => aoEvento(JSON.parse(e.data)))
    );
    return () => es.close();
  },
});
