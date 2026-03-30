import moviepy.editor as mp
import whisper
import os
import torch
import whisperx

def extrair_e_transcrever(caminho_video):
    nome_arquivo = os.path.splitext(caminho_video)[0]
    caminho_audio = f"{nome_arquivo}_extraido.mp3"

    # --- Passo 1: Extração de Áudio ---
    print(f"--- Extraindo áudio de: {caminho_video} ---")
    video = mp.VideoFileClip(caminho_video)
    video.audio.write_audiofile(caminho_audio)
    
    # --- Passo 2: Transcrição com Whisper ---
    print("\n--- Iniciando Transcrição (isso pode levar um tempo) ---")
    
    # Escolha do modelo: 'tiny', 'base', 'small', 'medium', 'large-v3'
    # O 'medium' ou 'large' são os melhores para Português.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    modelo = whisper.load_model("medium", device=device)
    
    # Executando a transcrição especificando o idioma
    resultado = modelo.transcribe(caminho_audio, language="pt", verbose=True)
    
    # --- Passo 3: Salvar o Resultado ---
    texto_final = resultado["text"]
    caminho_txt = f"{nome_arquivo}_transcricao.txt"
    
    with open(caminho_txt, "w", encoding="utf-8") as f:
        for segment in resultado['segments']:
            start = segment['start']
            end = segment['end']
            texto = segment['text']
            # Formata o tempo (ex: 00:01:30)
            timestamp = f"[{int(start//60):02d}:{int(start%60):02d}]"
            f.write(f"{timestamp} {texto}\n")
    
    print(f"\n✅ Concluído! Texto salvo em: {caminho_txt}")
    return texto_final

# Exemplo de uso
if __name__ == "__main__":
    arquivo_video = "G:/Meu Drive/ferramentas/transcriptAI/reuniao.mp4" # Substitua pelo seu arquivo
    if os.path.exists(arquivo_video):
        transcricao = extrair_e_transcrever(arquivo_video)
    else:
        print("Erro: Arquivo de vídeo não encontrado.")