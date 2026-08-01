// Sobe o backend Python como filho, lê o aperto de mão do stdout e abre a janela.
//
// O backend imprime UMA linha no stdout -- porta e token -- e nunca mais escreve ali.
// Todo o resto dele vai para stderr, que só encaminhamos para o console.

const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

const RAIZ = path.join(__dirname, "..");
const PYTHON = process.env.AXON_PYTHON || "python";

let backend = null;
let conexao = null; // {porta, token}
let janela = null;

function subirBackend() {
  return new Promise((resolve, reject) => {
    const args = ["-m", "axondesk", "--pai", String(process.pid)];
    if (process.env.AXON_APENAS) args.push("--apenas", process.env.AXON_APENAS);
    if (process.env.AXON_MODELO) args.push("--modelo", process.env.AXON_MODELO);

    backend = spawn(PYTHON, args, {
      cwd: path.join(RAIZ, "python"),
      // stdin ignorado de propósito: uma thread parada lendo stdin congela o backend
      // quando o pyaxon está carregado. A morte do pai é detectada por --pai.
      stdio: ["ignore", "pipe", "pipe"],
    });

    let buffer = "";
    const aoSair = setTimeout(
      () => reject(new Error("o backend não respondeu em 60s")),
      60000
    );

    backend.stdout.on("data", (d) => {
      buffer += d.toString();
      const linha = buffer.split("\n").find((l) => l.startsWith("AXON_DESK_READY"));
      if (linha) {
        clearTimeout(aoSair);
        conexao = JSON.parse(linha.slice("AXON_DESK_READY".length).trim());
        console.log("[backend] pronto na porta", conexao.porta);
        resolve(conexao);
      }
    });

    backend.stderr.on("data", (d) => process.stderr.write("[backend] " + d));
    backend.on("exit", (c) => console.log("[backend] saiu com", c));
    backend.on("error", reject);
  });
}

function criarJanela() {
  janela = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: "#14181c",
    title: "axon-desk",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  janela.loadFile(path.join(__dirname, "renderer", "index.html"));
}

ipcMain.handle("conexao", () => conexao);

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
