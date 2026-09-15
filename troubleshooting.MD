# Troubleshooting

Problemas reais enfrentados durante o desenvolvimento deste projeto, e como
foram resolvidos. Mantido como referência para builds futuros ou para quem
for reproduzir o ambiente.

---

## Docker build trava por muito tempo / "backtracking" do pip

**Sintoma:** `docker build` fica preso por dezenas de minutos numa etapa de
`pip install`, sem erro aparente, apenas sem progredir.

**Causa:** quando o `requirements.txt` lista pacotes sem versão fixada, o
`pip` moderno tenta resolver combinações compatíveis entre várias versões
possíveis ("backtracking"). Com um projeto com muitas dependências pesadas
(torch, opencv, matplotlib), essa busca pode ficar extremamente lenta.

**Solução:** fixar as versões exatas no `requirements.txt`, geradas a partir
de um ambiente já validado:
```bash
pip freeze > requirements.txt
```

---

## PyTorch instalando bibliotecas CUDA desnecessárias no Linux/Docker

**Sintoma:** o build do Docker baixa dezenas de pacotes `nvidia-*` e
`cuda-toolkit` (centenas de MB), mesmo a aplicação só fazendo inferência
em CPU.

**Causa:** o `pip install torch` a partir do índice padrão do PyPI, no
Linux, resolve por padrão para a build com suporte a GPU (CUDA), que traz
essas dependências extras.

**Solução:** instalar explicitamente a build CPU-only, usando o índice
oficial do PyTorch para CPU, **antes** de instalar o restante das
dependências, garantindo que ambas as instalações usem o mesmo
`site-packages` (sem `--prefix` separado, que faz o pip "esquecer" o
que já foi instalado):
```dockerfile
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
```

---

## Incompatibilidade de versão do Python entre ambiente local e Docker

**Sintoma:** erro do tipo `Could not find a version that satisfies the
requirement X==Y.Z / Requires-Python >=3.12` durante o build.

**Causa:** o `requirements.txt` foi gerado (via `pip freeze`) a partir de
um ambiente local rodando uma versão de Python diferente da usada na
imagem base do Dockerfile — algumas versões de pacote fixadas simplesmente
não existem para a versão de Python da imagem.

**Solução:** garantir que a imagem base do Dockerfile (`FROM python:X.Y-slim`)
use a **mesma versão major.minor** de Python do ambiente local onde o
`requirements.txt` foi gerado.

---

## Docker Desktop / WSL2 trava ou retorna "read-only file system"

**Sintoma:** comandos como `docker ps` ou `docker build` travam
indefinidamente, ou retornam erros de sistema de arquivos somente leitura.

**Causa:** builds muito longos (por causa do problema de backtracking acima)
podem esgotar o disco virtual usado pelo WSL2, corrompendo o estado interno
do Docker Desktop no Windows.

**Solução, em ordem de intensidade:**
1. Limpar cache não utilizado: `docker system prune -a --volumes -f`
2. Reset das distribuições WSL2 do Docker:
   ```powershell
   wsl --shutdown
   wsl --unregister docker-desktop
   wsl --unregister docker-desktop-data
   ```
3. Se nada resolver, reiniciar a máquina por completo — processos do WSL2
   podem ficar presos na memória mesmo após tentativas de encerramento manual.

**Prevenção:** monitorar `docker system df` periodicamente e limpar cache
acumulado de builds antigos, especialmente após builds que falharam no meio.

---

## `curl` no PowerShell não aceita a mesma sintaxe do curl Unix

**Sintoma:** erro `Não é possível localizar um parâmetro que coincida com
o nome de parâmetro 'X'` ao rodar comandos `curl -X POST ...`.

**Causa:** o PowerShell define `curl` como um alias para `Invoke-WebRequest`,
que tem sintaxe de parâmetros diferente do `curl` real (Unix).

**Solução:** chamar o executável real explicitamente com `curl.exe` em vez
de `curl`, ou usar a sintaxe nativa do PowerShell (`Invoke-RestMethod`).

---

## `mkdir -p` e `touch` não existem no PowerShell

**Sintoma:** `mkdir : Não é possível localizar um parâmetro posicional`
ou `touch não é reconhecido como nome de cmdlet`.

**Causa:** esses são comandos do shell Unix (Bash), não existem nativamente
no PowerShell.

**Solução:**
```powershell
# Criar múltiplas pastas
mkdir pasta1, pasta2, pasta3

# Criar arquivo vazio (equivalente ao touch)
New-Item -ItemType File -Path "arquivo.py" -Force
```