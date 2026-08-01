// Duas aplicações, um main. Qual abre vem de `--painel=chat|code` (ou AXON_PAINEL).
//
// Cada aplicação sobe o próprio backend. Podia haver um só compartilhado, mas isso
// exigiria descoberta e ciclo de vida entre processos -- e com um expert a subida é de
// segundos. Quando os 19 entrarem, vale compartilhar.

const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

const RAIZ = path.join(__dirname, "..");
const PYTHON = process.env.AXON_PYTHON || "python";

const PAINEL =
  (process.argv.find((a) => a.startsWith("--painel=")) || "").split("=")[1] ||
  process.env.AXON_PAINEL ||
  "chat";

const JANELAS = {
  chat: { titulo: "axon-chat", largura: 900, altura: 820 },
  code: { titulo: "axon-code", largura: 940, altura: 640 },
};

let backend = null;
let conexao = null;

function subirBackend() {
  return new Promise((resolve, reject) => {
    const args = ["-m", "axondesk", "--pai", String(process.pid)];
    if (process.env.AXON_APENAS) args.push("--apenas", process.env.AXON_APENAS);
    if (process.env.AXON_MODELO) args.push("--modelo", process.env.AXON_MODELO);

    backend = spawn(PYTHON, args, {
      cwd: path.join(RAIZ, "python"),
      // stdin ignorado: uma thread parada lendo stdin congela o backend com o pyaxon
      // carregado. A morte do pai é detectada por --pai.
      stdio: ["ignore", "pipe", "pipe"],
    });

    let buffer = "";
    const prazo = setTimeout(
      () => reject(new Error("o backend não respondeu em 90s")),
      90000
    );

    backend.stdout.on("data", (d) => {
      buffer += d.toString();
      const linha = buffer
        .split("\n")
        .find((l) => l.startsWith("AXON_DESK_READY"));
      if (linha) {
        clearTimeout(prazo);
        conexao = JSON.parse(linha.slice("AXON_DESK_READY".length).trim());
        console.log(`[${PAINEL}] backend na porta`, conexao.porta);
        resolve(conexao);
      }
    });

    backend.stderr.on("data", (d) => process.stderr.write(`[${PAINEL}] ` + d));
    backend.on("error", reject);
  });
}

function criarJanela() {
  const cfg = JANELAS[PAINEL] || JANELAS.chat;
  const janela = new BrowserWindow({
    width: cfg.largura,
    height: cfg.altura,
    backgroundColor: PAINEL === "code" ? "#10141a" : "#14181c",
    title: cfg.titulo,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  janela.loadFile(path.join(__dirname, PAINEL, "index.html"));
}

ipcMain.handle("conexao", () => conexao);
ipcMain.handle("painel", () => PAINEL);

app.whenReady().then(async () => {
  try {
    await subirBackend();
  } catch (e) {
    console.error("falha ao subir o backend:", e.message);
  }
  criarJanela();
});

app.on("window-all-closed", () => app.quit());
app.on("before-quit", () => {
  if (backend && !backend.killed) backend.kill();
});
