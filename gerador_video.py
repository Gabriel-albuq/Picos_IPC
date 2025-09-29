import cv2
import numpy as np
import os

def criar_video_com_imagem(caminho_imagem, nome_video_saida, duracao_total_s=5, duracao_preto_s=2, fps=30):
    """
    Cria um vídeo com uma tela preta inicial seguida por uma imagem estática.

    Args:
        caminho_imagem (str): O caminho para a imagem de entrada.
        nome_video_saida (str): O nome do arquivo de vídeo a ser criado (ex: 'video.mp4').
        duracao_total_s (int): Duração total do vídeo em segundos.
        duracao_preto_s (int): Duração da tela preta inicial em segundos.
        fps (int): Frames por segundo do vídeo.
    """
    # Verifica se a imagem de entrada existe
    if not os.path.exists(caminho_imagem):
        print(f"Erro: O arquivo de imagem '{caminho_imagem}' não foi encontrado.")
        return

    # 1. Carregar a imagem
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"Erro: Não foi possível ler o arquivo de imagem '{caminho_imagem}'. Verifique se é um formato válido.")
        return
        
    altura, largura, _ = imagem.shape
    tamanho_frame = (largura, altura)

    # 2. Configurar o escritor de vídeo
    # O codec 'mp4v' é uma boa opção para arquivos .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(nome_video_saida, fourcc, fps, tamanho_frame)

    # 3. Calcular o número de frames para cada parte
    frames_pretos = int(fps * duracao_preto_s)
    frames_imagem = int(fps * (duracao_total_s - duracao_preto_s))

    # Criar um frame preto com as mesmas dimensões da imagem
    frame_preto = np.zeros((altura, largura, 3), dtype=np.uint8)

    print(f"Gerando {duracao_preto_s}s de tela preta...")
    # 4. Escrever os frames pretos no vídeo
    for _ in range(frames_pretos):
        video_writer.write(frame_preto)

    print(f"Adicionando a imagem por {duracao_total_s - duracao_preto_s}s...")
    # 5. Escrever os frames da imagem no vídeo
    for _ in range(frames_imagem):
        video_writer.write(imagem)

    # 6. Liberar o objeto de vídeo para salvar o arquivo
    video_writer.release()
    print(f"\nVídeo '{nome_video_saida}' criado com sucesso!")
    print(f"Resolução: {largura}x{altura} | Duração: {duracao_total_s}s | FPS: {fps}")


# --- INÍCIO DA EXECUÇÃO ---
if __name__ == "__main__":
    # Coloque o nome da imagem que você enviou aqui
    nome_da_sua_imagem = "data/outputs/novo_teste/ORIGINAL/SM/SM_14_original__20250929_124558.jpg"
    
    # Nome do arquivo de vídeo que será gerado
    video_de_saida = "video_gerado.mp4"
    
    criar_video_com_imagem(nome_da_sua_imagem, video_de_saida)
