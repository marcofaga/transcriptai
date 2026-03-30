import moviepy.editor as mp
import whisperx
import os
import torch
import gc
from dotenv import load_dotenv

def extrair_e_transcrever_com_diarizacao(caminho_video, hf_token):
    nome_arquivo = os.path.splitext(caminho_video)[0]
    caminho_audio = f"{nome_arquivo}_extraido.mp3"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 
    compute_type = "float16" # Ideal para sua RTX 4060 Ti

    # --- Passo 1: Extração de Áudio ---
    if not os.path.exists(caminho_audio):
        print(f"--- Extraindo áudio de: {caminho_video} ---")
        video = mp.VideoFileClip(caminho_video)
        video.audio.write_audiofile(caminho_audio)
    
    audio = whisperx.load_audio(caminho_audio)

    # --- Passo 2: Transcrição (Modelo Large-v2 conforme tutorial) ---
    print(f"\n--- Iniciando Transcrição no {device} (Modelo Large-v2) ---")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    # Removido language="pt" fixo para seguir o fluxo do tutorial, 
    # ou mantido se você tiver certeza que é sempre português.
    resultado = model.transcribe(audio, batch_size=batch_size)

    # Limpeza de memória GPU após transcrição
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Passo 3: Alinhamento ---
    print("--- Alinhando timestamps ---")
    # O tutorial sugere usar result["language"] detectado automaticamente
    model_a, metadata = whisperx.load_align_model(language_code=resultado["language"], device=device)
    resultado = whisperx.align(resultado["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Limpeza de memória GPU após alinhamento
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # --- Passo 4: Diarização ---
    print("--- Identificando interlocutores ---")
    # No tutorial oficial, a classe é chamada via whisperx.diarize.DiarizationPipeline
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    
    # Executa a diarização
    diarize_segments = diarize_model(audio)
    
    # Une locutores ao texto (passando o resultado completo como no tutorial)
    resultado_final = whisperx.assign_word_speakers(diarize_segments, resultado)

    # --- Passo 5: Salvar o Resultado ---
    caminho_txt = f"{nome_arquivo}_transcricao_v2.txt"
    
    with open(caminho_txt, "w", encoding="utf-8") as f:
        for segment in resultado_final['segments']:
            start = segment['start']
            speaker = segment.get('speaker', 'DESCONHECIDO')
            texto = segment['text'].strip()
            
            timestamp = f"[{int(start//60):02d}:{int(start%60):02d}]"
            linha = f"{timestamp} ({speaker}): {texto}\n"
            f.write(linha)
            print(linha, end="")
    
    print(f"\n✅ Concluído! Texto salvo em: {caminho_txt}")
    return resultado_final

if __name__ == "__main__":
    # Importante: Removi seu token real do código por segurança. 
    # Substitua pela sua chave novamente.
    MEU_TOKEN_HF = os.getenv("HF_TOKEN")
    
    arquivo_video = "/mnt/c/github/transcriptAI/reuniao.mp4"
    
    if os.path.exists(arquivo_video):
        extrair_e_transcrever_com_diarizacao(arquivo_video, MEU_TOKEN_HF)
    else:
        print(f"Erro: Arquivo de vídeo não encontrado em {arquivo_video}")