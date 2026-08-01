// Os dois painéis leem o MESMO fluxo de eventos e cada um mostra o que lhe cabe.
// É essa origem única que faz o comando rodado no axon-code aparecer no contexto do
// axon-chat depois -- e o que justifica existirem dois painéis em vez de um.

const $ = (id) => document.getElementById(id);

const conversa = $("conversa");
const terminal = $("terminal");
let cwd = null;
let corrente = null; // {run_id, elemento}

// ---------------------------------------------------------------- utilidades
function rolar(el) {
  el.scrollTop = el.scrollHeight;
}

function bloco(pai, classe, quem, texto) {
  const d = document.createElement("div");
  d.className = "msg " + classe;
  if (quem) {
    const q = document.createElement("div");
    q.className = "quem";
    q.textContent = quem;
    d.appendChild(q);
  }
  const c = document.createElement("div");
  c.className = "corpo";
  c.textContent = texto || "";
  d.appendChild(c);
  pai.appendChild(d);
  rolar(pai);
  return c;
}

function linhaTerminal(texto, classe) {
  const d = document.createElement("div");
  if (classe) d.className = classe;
  d.textContent = texto;
  terminal.appendChild(d);
  rolar(terminal);
  return d;
}

// ---------------------------------------------------------------- estado
async function atualizarEstado() {
  try {
    const s = await window.axon.saude();
    $("e-backend").textContent = "no ar";
    $("e-backend").className = "ok";
    const e = s.experts || {};
    $("e-experts").textContent =
      e.estado === "pronto"
        ? `${(e.carregados || []).length} pronto(s)`
        : `${(e.carregados || []).length}/${e.total || "?"}`;
    $("e-experts").className = e.estado === "pronto" ? "ok" : "";
    $("e-ollama").textContent = s.ollama ? "no ar" : "fora";
    $("e-ollama").className = s.ollama ? "ok" : "ruim";
    $("e-modelo").textContent = (s.modelos || [])[0] || "—";
  } catch {
    $("e-backend").textContent = "fora";
    $("e-backend").className = "ruim";
  }
}

// ---------------------------------------------------------------- eventos
function aoEvento(ev) {
  switch (ev.tipo) {
    case "boot":
      atualizarEstado();
      break;

    case "usuario":
      // já desenhado no envio; ignorado para não duplicar
      break;

    case "rota": {
      const d = document.createElement("div");
      d.className = "rota";
      const cam = (ev.passagens || [])
        .slice(0, 2)
        .map((p) => p.caminho.join(" > "))
        .join(" · ");
      d.innerHTML = `<b>${ev.expert}</b>${cam ? " — " + cam : ""}`;
      conversa.appendChild(d);
      $("rota-atual").textContent = ev.expert;
      corrente = { run_id: ev.run_id, elemento: null };
      rolar(conversa);
      break;
    }

    case "delta":
      if (!corrente || !corrente.elemento) {
        const p = conversa.querySelector(".pensando");
        if (p) p.remove();
        corrente = corrente || {};
        corrente.elemento = bloco(conversa, "ia", "axon", "");
      }
      corrente.elemento.textContent += ev.texto;
      rolar(conversa);
      break;

    case "fim": {
      $("t-entrada").textContent = ev.tokens_entrada;
      $("t-saida").textContent = ev.tokens_saida;
      const s = (ev.ms_prefill / 1000).toFixed(1);
      $("t-prefill").textContent = `${s}s`;
      terminarEnvio();
      break;
    }

    case "cancelado":
      bloco(conversa, "erro", null, "cancelado");
      terminarEnvio();
      break;

    case "erro":
      bloco(conversa, "erro", "erro", ev.erro);
      terminarEnvio();
      break;

    // ------------------------------------------------------- axon-code
    case "comando":
      cwd = ev.cwd;
      $("cwd").textContent = ev.cwd;
      linhaTerminal(ev.comando, "cmd");
      break;

    case "saida":
      linhaTerminal(ev.linha);
      break;

    case "comando_fim":
      linhaTerminal(
        ev.estourou
          ? `— tempo esgotado (${ev.timeout}s)`
          : `— saiu com ${ev.codigo}`,
        "fim" + (ev.codigo === 0 ? "" : " ruim")
      );
      break;
  }
}

// ---------------------------------------------------------------- chat
function terminarEnvio() {
  corrente = null;
  $("enviar").disabled = false;
  $("enviar").hidden = false;
  $("parar").hidden = true;
  const p = conversa.querySelector(".pensando");
  if (p) p.remove();
}

$("form-chat").addEventListener("submit", async (e) => {
  e.preventDefault();
  const campo = $("campo-chat");
  const texto = campo.value.trim();
  if (!texto) return;

  const vazio = conversa.querySelector(".vazio");
  if (vazio) vazio.remove();

  bloco(conversa, "eu", "você", texto);
  campo.value = "";
  campo.style.height = "auto";

  $("enviar").hidden = true;
  $("parar").hidden = false;

  const d = document.createElement("div");
  d.className = "pensando";
  d.textContent = "roteando e recuperando…";
  conversa.appendChild(d);
  rolar(conversa);

  const r = await window.axon.perguntar(texto, $("e-modelo").textContent);
  corrente = { run_id: r.run_id, elemento: null };
});

$("parar").addEventListener("click", async () => {
  if (corrente && corrente.run_id) await window.axon.cancelar(corrente.run_id);
});

// Enter envia; Shift+Enter quebra linha.
$("campo-chat").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("form-chat").requestSubmit();
  }
});
$("campo-chat").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 128) + "px";
});

// ---------------------------------------------------------------- axon-code
$("form-code").addEventListener("submit", async (e) => {
  e.preventDefault();
  const campo = $("campo-code");
  const cmd = campo.value.trim();
  if (!cmd) return;
  campo.value = "";
  await window.axon.executar(cmd, cwd);
});

// ---------------------------------------------------------------- partida
window.axon.ouvir(aoEvento);
atualizarEstado();
setInterval(atualizarEstado, 5000);
