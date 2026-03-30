# 🎙️ TranscriptAI com WhisperX e Diarização

WhisperX Transcritor & Diarização: Transcrição de áudio de alta precisão com identificação de locutores utilizando OpenAI Whisper (Large-v2), Pyannote e aceleração CUDA. Otimizado para alinhamento rápido de áudio e diarização de fala em GPUs NVIDIA.

Este projeto automatiza a extração de áudio, transcrição e **identificação de locutores (diarização)** a partir de arquivos de vídeo. Ele utiliza o modelo **WhisperX**, otimizado para rodar em GPUs NVIDIA via CUDA, garantindo alta velocidade e alinhamento preciso de timestamps.

## 🚀 Funcionalidades

* **Extração Automática:** Converte vídeos em áudio MP3 para processamento.
* **Transcrição de Alta Precisão:** Utiliza o modelo `large-v2` do WhisperX.
* **Alinhamento de Fonemas:** Melhora a precisão dos timestamps por palavra.
* **Diarização de Locutores:** Identifica quem falou o quê no diálogo.
* **Gestão de Memória:** Implementa limpeza de cache da GPU entre etapas para evitar erros de memória em placas como a RTX 4060 Ti.

---

## 🛠️ Requisitos de Sistema

* **SO:** Linux ou Windows (via WSL2 recomendado).
* **GPU:** NVIDIA com **CUDA 12.8** instalado.
* **Python:** 3.10 ou superior.
* **FFmpeg:** Instalado no sistema.

---

## 📦 Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/marcofaga/transcriptai.git
   cd seu-repositorio
   ```

2. **Crie um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/WSL
   ```

3. **Instale as dependências:**
   O instalador do WhisperX cuidará da versão correta do PyTorch compatível com sua GPU.
   ```bash
   pip install whisperx moviepy==1.0.3 python-dotenv
   sudo apt update && sudo apt install ffmpeg -y
   ```

---

## 🔐 Configuração do Token Hugging Face

Para a diarização, é necessário aceitar os termos de uso dos modelos da Pyannote no Hugging Face:

1. Aceite os termos em: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) e [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
2. Crie um token de acesso em *Settings -> Access Tokens*.
3. Crie um arquivo chamado **`.env`** na raiz do projeto:
   ```text
   HF_TOKEN=seu_token_aqui
   ```

---

## 💻 Como Usar

1. Coloque o caminho do seu vídeo na variável `arquivo_video` dentro do bloco `if __name__ == "__main__":` no arquivo principal.
2. Execute o script:
   ```bash
   python nome_do_seu_script.py
   ```
3. O resultado será gerado em um arquivo `.txt` com o mesmo nome do vídeo original, formatado da seguinte forma:
   `[MM:SS] (SPEAKER_00): Texto transcrito aqui.`

---

## ⚠️ Notas Técnicas (GPU/CUDA)

* O código está configurado para `compute_type="float16"`, que é o ideal para GPUs modernas (RTX Série 40), equilibrando velocidade e precisão.
* Se encontrar erros de compatibilidade com o PyTorch, certifique-se de não ter instalado versões manuais do Torch antes do WhisperX.

---

## 📄 Licença

Este projeto está sob a licença GPL-3.0. Veja o arquivo [LICENSE](LICENSE) para detalhes.