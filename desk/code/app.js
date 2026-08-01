// axon-code: um terminal que também responde.
//
// Quem decide se a linha é comando ou pergunta é o backend, não esta tela -- assim a
// mesma regra vale para qualquer cliente. `git status` executa em milissegundos; o que
// não for executável vai para o expert.

const $ = (id) => document.getElementById(id);
const terminal = $("terminal");

let cwd = null;
let modelo = null;
let corrente = null;
const historico = [];
let pos = -1;

const rolar = () => (terminal.scrollTop = terminal.scrollHeight);

function linha(texto, classe) {
  const d = document.createElement("div");
  if (classe) d.className = classe;
  d.textContent = texto;
  terminal.appendChild(d);
  rolar();
  return d;
}

async function atualizarEstado() {
  try {
    const s = await window.axon.saude();
    const e = s.experts || {};
    $("e-experts").textContent =
      e.estado === "pronto"
        ? `${(e.carregados || []).length} expert(s)`
        : `carregando ${(e.carregados || []).length}/${e.total || "?"}`;
    $("e-experts").className = e.estado === "pronto" ? "ok" : "";
    $("e-ollama").textContent = s.ollama ? "ollama no ar" : "ollama fora";
    $("e-ollama").className = s.ollama ? "ok" : "ruim";
    modelo = (s.modelos || [])[0] || null;
  } catch {
    $("e-experts").textContent = "backend fora";
    $("e-experts").className = "ruim";
  }
}

function terminar() {
  corrente = null;
  $("parar").hidden = true;
  $("campo").disabled = false;
  $("campo").focus();
}

window.axon.ouvir((ev) => {
  switch (ev.tipo) {
    case "boot":
      atualizarEstado();
      break;

    case "comando":
      cwd = ev.cwd;
      $("cwd").textContent = ev.cwd;
      break;

    case "saida":
      linha(ev.linha);
      break;

    case "comando_fim":
      linha(
        ev.estourou
          ? `— tempo esgotado (${ev.timeout}s)`
          : `— saiu com ${ev.codigo}`,
        "fim" + (ev.codigo === 0 ? "" : " ruim")
      );
      terminar();
      break;

    // ------------------------------------------------- resposta do modelo
    case "rota":
      linha(`⟨${ev.expert}⟩ recuperando…`, "rota-linha");
      break;

    case "delta":
      if (!corrente || !corrente.elemento) {
        corrente = corrente || {};
        corrente.elemento = linha("", "resposta");
      }
      corrente.elemento.textContent += ev.texto;
      rolar();
      break;

    case "fim":
      linha(
        `— ${ev.tokens_entrada}/${ev.tokens_saida} tokens · ` +
          `${(ev.ms_prefill / 1000).toFixed(0)}s de prefill`,
        "fim"
      );
      terminar();
      break;

    case "cancelado":
      linha("— cancelado", "fim ruim");
      terminar();
      break;

    case "erro":
      linha("— " + ev.erro, "fim ruim");
      terminar();
      break;
  }
});

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const campo = $("campo");
  const texto = campo.value.trim();
  if (!texto || corrente) return;

  historico.push(texto);
  pos = historico.length;
  linha(texto, "cmd");
  campo.value = "";

  // Comandos da própria tela, sem ir ao backend.
  if (texto === ":limpar") {
    terminal.innerHTML = "";
    return;
  }
  if (texto.startsWith(":cd ")) {
    cwd = texto.slice(4).trim();
    $("cwd").textContent = cwd;
    linha(`cwd = ${cwd}`, "fim");
    return;
  }

  campo.disabled = true;
  $("parar").hidden = false;
  const r = await window.axon.linha(texto, cwd, modelo);
  corrente = { run_id: r.run_id, elemento: null };
  if (r.modo === "pergunta") linha("(pergunta — isto demora em CPU)", "fim");
});

$("parar").addEventListener("click", async () => {
  if (corrente && corrente.run_id) await window.axon.cancelar(corrente.run_id);
});

// Seta pra cima e pra baixo percorrem o histórico, como num terminal de verdade.
$("campo").addEventListener("keydown", (e) => {
  if (e.key === "ArrowUp" && pos > 0) {
    e.preventDefault();
    $("campo").value = historico[--pos];
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    $("campo").value = ++pos < historico.length ? historico[pos] : "";
    if (pos > historico.length) pos = historico.length;
  }
});

atualizarEstado();
setInterval(atualizarEstado, 5000);
$("campo").focus();
