import os
import sys
import torch
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
print(root_path)

from src.func import (
    device_config,
    device_start,
    device_start_capture,
    load_model,
    start_application_interface
)

def load_settings(config_path):
    """Função para ler as configurações de um arquivo .txt e atribuir os valores diretamente às variáveis."""
    # Valores padrão
    defaults = {
        'perc_top': 0.4,
        'perc_bottom': 0.8,
        'min_score': 0.5,
        'limit_center': 8,
        'save_dir': 'data\\outputs\\capturas',
        'square_size': 640,
        'grid_x': 0,
        'grid_y': 0,
        'crop_image': 1  # 1 = Sim
    }

    try:
        with open(config_path, 'r') as file:
            for linha in file:
                # Ignora linhas vazias e comentários
                if linha.strip() and not linha.startswith('#'):
                    try:
                        chave, valor = linha.split('=', 1)
                        chave = chave.strip()
                        valor = valor.strip()
                        
                        # Só processa se o valor não for 'None'
                        if valor != 'None':
                            if chave in defaults:
                                # Converte para o tipo apropriado
                                if isinstance(defaults[chave], float):
                                    defaults[chave] = float(valor)
                                elif isinstance(defaults[chave], int):
                                    defaults[chave] = int(valor)
                                elif isinstance(defaults[chave], str):
                                    defaults[chave] = valor
                    except ValueError:
                        # Se houver erro na conversão, mantém o valor padrão
                        continue
    except FileNotFoundError:
        print(f'Arquivo de configuração não encontrado, usando valores padrão.')
    except Exception as e:
        print(f'Erro ao ler o arquivo: {e}. Usando valores padrão.')

    return (
        defaults['perc_top'],
        defaults['perc_bottom'],
        defaults['min_score'],
        defaults['limit_center'],
        defaults['save_dir'],
        defaults['square_size'],
        defaults['grid_x'],
        defaults['grid_y'],
        defaults['crop_image'],
    )


def save_settings(config_path, perc_top, 
                 perc_bottom, min_score, limit_center, 
                 save_dir, crop_image, square_size, grid_x, grid_y):
    """Função para salvar as configurações atuais diretamente nas variáveis no arquivo .txt."""
    arquivo = config_path
    try:
        with open(arquivo, 'w') as file:
            file.write(f'perc_top = {perc_top}\n')
            file.write(f'perc_bottom = {perc_bottom}\n')
            file.write(f'min_score = {min_score}\n')
            file.write(f'limit_center = {limit_center}\n')
            file.write(f'square_size = {square_size}\n')
            file.write(f'grid_x = {grid_x}\n')
            file.write(f'grid_y = {grid_y}\n')
            file.write(f'crop_image = {crop_image}\n')
            
            # Verifica se o save_dir não é None antes de salvar
            if save_dir is not None:
                file.write(f'save_dir = {save_dir}\n')

        print('\nConfigurações salvas com sucesso.\n')
    except Exception as e:
        print(f'\nErro ao salvar as configurações: {e}\n')


def start_application_interface(config_path):
    # Dicionário para armazenar os resultados que serão retornados
    result = {
        'linha': None,
        'device_name': None,
        'device_path': None,
        'option_visualize': None,
        'perc_top': None,
        'perc_bottom': None,
        'min_score': None,
        'limit_center': None,
        'save_dir': None,
        'square_size': None,
        'grid_x': None,
        'grid_y': None,
        'camera_backend': None,  # Novo campo para o backend da câmera
        'crop_image': None  # Novo campo para cortar imagem
    }

    (
        perc_top,
        perc_bottom,
        min_score,
        limit_center,
        save_dir,
        square_size,
        grid_x,
        grid_y,
        crop_image,
    ) = load_settings(config_path)
    
    def submit():
        # Validação dos campos obrigatórios
        if not linha_entry.get():
            messagebox.showerror("Erro", "O campo 'Linha' não pode estar vazio.")
            return
        if not device_name_entry.get():
            messagebox.showerror("Erro", "O campo 'Nome da Câmera/Vídeo' não pode estar vazio.")
            return
        if not device_path_var.get():
            messagebox.showerror("Erro", "O campo 'Câmera/Vídeo' não pode estar vazio.")
            return
        if not camera_backend_var.get():
            messagebox.showerror("Erro", "Você deve selecionar um backend para a câmera (OpenCV ou GxCam).")
            return
        if not square_size_entry.get():
            messagebox.showerror("Erro", "O campo 'Tamanho da área de interesse' não pode estar vazio.")
            return
        if not grid_x_entry.get():
            messagebox.showerror("Erro", "O campo 'Localização em X' não pode estar vazio.")
            return
        if not grid_y_entry.get():
            messagebox.showerror("Erro", "O campo 'Localização em Y' não pode estar vazio.")
            return
        if not perc_top_entry.get():
            messagebox.showerror("Erro", "O campo 'Percentual Mínimo' não pode estar vazio.")
            return
        if not perc_bottom_entry.get():
            messagebox.showerror("Erro", "O campo 'Percentual Máximo' não pode estar vazio.")
            return
        if not min_score_entry.get():
            messagebox.showerror("Erro", "O campo 'Score Mínimo' não pode estar vazio.")
            return
        if not limit_center_entry.get():
            messagebox.showerror("Erro", "O campo 'Limite de centro' não pode estar vazio.")
            return

        # Preenche o dicionário result com os valores dos campos
        result['linha'] = linha_entry.get()
        result['device_name'] = device_name_entry.get()
        result['device_path'] = device_path_var.get()
        result['option_visualize'] = int(option_var.get())
        result['perc_top'] = float(perc_top_entry.get())
        result['perc_bottom'] = float(perc_bottom_entry.get())
        result['min_score'] = float(min_score_entry.get())
        result['limit_center'] = int(limit_center_entry.get())
        result['square_size'] = int(square_size_entry.get())
        result['grid_x'] = int(grid_x_entry.get())
        result['grid_y'] = int(grid_y_entry.get())
        result['camera_backend'] = camera_backend_var.get()
        result['crop_image'] = int(crop_image_var.get())
        
        # Verifica a opção de salvar detecções
        if not save_detection_var.get():  # Caso "Não salvar detecções" esteja marcado
            result['save_dir'] = save_dir_var.get()
        else:
            result['save_dir'] = None
            
        root.destroy()  # Fecha a janela após obter os valores
    
    def browse_file():
        filename = filedialog.askopenfilename()
        if filename:
            device_path_var.set(filename)
    
    def browse_save_dir():
        directory = filedialog.askdirectory()
        if directory:
            save_dir_var.set(directory)
    
    def toggle_save_dir(*args):
        """Habilita ou desabilita o campo de diretório de salvar baseado no checkbox."""
        if save_detection_var.get():
            save_dir_entry.config(state="disabled")  # Desabilita o campo de diretório
        else:
            save_dir_entry.config(state="normal")  # Habilita o campo de diretório
    
    root = tk.Tk()
    root.title("Configuração do PICOS")
    
    # Configuração do layout
    pad_x = 10
    pad_y = 5
    
    # Entrada para a linha
    tk.Label(root, text="Digite a linha:", anchor='w', width=30).grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky='w')
    linha_entry = tk.Entry(root, width=30)
    linha_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y)
    
    # Entrada para o nome da câmera/vídeo
    tk.Label(root, text="Nome da Câmera/Vídeo:", anchor='w', width=30).grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky='w')
    device_name_entry = tk.Entry(root, width=30)
    device_name_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y)
    
    # Entrada para o caminho do dispositivo
    tk.Label(root, text="Câmera/Vídeo:", anchor='w', width=30).grid(row=2, column=0, padx=pad_x, pady=pad_y, sticky='w')
    device_path_var = tk.StringVar()
    device_path_entry = tk.Entry(root, textvariable=device_path_var, width=30)
    device_path_entry.grid(row=2, column=1, padx=pad_x, pady=pad_y)
    
    # Botão para procurar um arquivo de vídeo
    tk.Button(root, text="Procurar", command=browse_file, width=10).grid(row=2, column=2, padx=pad_x, pady=pad_y)
    
    # Seleção de backend da câmera
    tk.Label(root, text="Backend da Câmera:", anchor='w', width=30).grid(row=3, column=0, padx=pad_x, pady=pad_y, sticky='w')
    camera_backend_var = tk.StringVar(value="")
    frame_backend = tk.Frame(root)
    frame_backend.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky='w')
    tk.Radiobutton(frame_backend, text="OpenCV", variable=camera_backend_var, value="OpenCV").pack(side='left')
    tk.Radiobutton(frame_backend, text="GxCam", variable=camera_backend_var, value="GxCam").pack(side='left')
    
    # Opção de visualização
    tk.Label(root, text="Visualizar predições:", anchor='w', width=30).grid(row=4, column=0, padx=pad_x, pady=pad_y, sticky='w')
    option_var = tk.StringVar(value=1)
    frame_options = tk.Frame(root)
    frame_options.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky='w')
    tk.Radiobutton(frame_options, text="Sim", variable=option_var, value="1").pack(side='left')
    tk.Radiobutton(frame_options, text="Não", variable=option_var, value="0").pack(side='left')
    
    # Diretório de salvar
    tk.Label(root, text="Salvar detecções em:", anchor='w', width=30).grid(row=5, column=0, padx=pad_x, pady=pad_y, sticky='w')
    save_dir_var = tk.StringVar()
    save_dir_entry = tk.Entry(root, textvariable=save_dir_var, width=30)
    save_dir_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y)

    save_dir_var.set(save_dir)  # Preenche com o valor do config.txt
    tk.Button(root, text="Procurar", command=browse_save_dir, width=10).grid(row=5, column=2, padx=pad_x, pady=pad_y)

    # Caminho padrão abaixo do campo
    tk.Label(root, text="Caminho Padrão: data\\outputs\\capturas", anchor='w', width=30).grid(row=6, column=1, padx=pad_x, pady=pad_y, sticky='w')

    # Opção para "Não salvar detecções" (agora com valor padrão marcado)
    save_detection_var = tk.BooleanVar(value=True)  # Agora inicia como True (marcado)
    save_detection_checkbox = tk.Checkbutton(root, text="Não salvar detecções", variable=save_detection_var, command=toggle_save_dir)
    save_detection_checkbox.grid(row=7, column=1, padx=pad_x, pady=pad_y, sticky='w')

    # Opção para "Cortar imagem?" (novo campo)
    tk.Label(root, text="Cortar imagem:", anchor='w', width=30).grid(row=8, column=0, padx=pad_x, pady=pad_y, sticky='w')
    crop_image_var = tk.StringVar(value=1)  # Valor padrão 1 (Sim)
    frame_crop = tk.Frame(root)
    frame_crop.grid(row=8, column=1, padx=pad_x, pady=pad_y, sticky='w')
    tk.Radiobutton(frame_crop, text="Sim", variable=crop_image_var, value="1").pack(side='left')
    tk.Radiobutton(frame_crop, text="Não", variable=crop_image_var, value="0").pack(side='left')

    # Tamanho da área de interesse
    tk.Label(root, text="Tamanho da área de interesse:", anchor='w', width=30).grid(row=9, column=0, padx=pad_x, pady=pad_y, sticky='w')
    square_size_entry = tk.Entry(root, width=30)
    square_size_entry.grid(row=9, column=1, padx=pad_x, pady=pad_y)
    square_size_entry.insert(0, square_size)  # Preenche com o valor do config.txt
    
    # Localização em X
    tk.Label(root, text="Localização em X:", anchor='w', width=30).grid(row=10, column=0, padx=pad_x, pady=pad_y, sticky='w')
    grid_x_entry = tk.Entry(root, width=30)
    grid_x_entry.grid(row=10, column=1, padx=pad_x, pady=pad_y)
    grid_x_entry.insert(0, grid_x)  # Preenche com o valor do config.txt
    
    # Localização em Y
    tk.Label(root, text="Localização em Y:", anchor='w', width=30).grid(row=11, column=0, padx=pad_x, pady=pad_y, sticky='w')
    grid_y_entry = tk.Entry(root, width=30)
    grid_y_entry.grid(row=11, column=1, padx=pad_x, pady=pad_y)
    grid_y_entry.insert(0, grid_y)  # Preenche com o valor do config.txt

    # Parâmetros adicionais
    # Percentual mínimo
    tk.Label(root, text="Percentual Mínimo:", anchor='w', width=30).grid(row=12, column=0, padx=pad_x, pady=pad_y, sticky='w')
    perc_top_entry = tk.Entry(root, width=30)
    perc_top_entry.grid(row=12, column=1, padx=pad_x, pady=pad_y)
    perc_top_entry.insert(0, perc_top)  # Preenche com o valor do config.txt
    
    # Percentual máximo
    tk.Label(root, text="Percentual Máximo:", anchor='w', width=30).grid(row=13, column=0, padx=pad_x, pady=pad_y, sticky='w')
    perc_bottom_entry = tk.Entry(root, width=30)
    perc_bottom_entry.grid(row=13, column=1, padx=pad_x, pady=pad_y)
    perc_bottom_entry.insert(0, perc_bottom)  # Preenche com o valor do config.txt
    
    # Score mínimo
    tk.Label(root, text="Score Mínimo:", anchor='w', width=30).grid(row=14, column=0, padx=pad_x, pady=pad_y, sticky='w')
    min_score_entry = tk.Entry(root, width=30)
    min_score_entry.grid(row=14, column=1, padx=pad_x, pady=pad_y)
    min_score_entry.insert(0, min_score)  # Preenche com o valor do config.txt
    
    # Limite de centro
    tk.Label(root, text="Limite de centro:", anchor='w', width=30).grid(row=15, column=0, padx=pad_x, pady=pad_y, sticky='w')
    limit_center_entry = tk.Entry(root, width=30)
    limit_center_entry.grid(row=15, column=1, padx=pad_x, pady=pad_y)
    limit_center_entry.insert(0, limit_center)  # Preenche com o valor do config.txt
    
    # Botão de confirmação
    tk.Button(root, text="Confirmar", command=submit, width=20).grid(row=16, column=0, columnspan=4, pady=10)
    
    # Chama a função para ajustar o estado do diretório de salvar
    toggle_save_dir()
    
    root.mainloop()
    
    # Salva as configurações
    save_settings(config_path, 
                result['perc_top'], 
                result['perc_bottom'],
                result['min_score'], 
                result['limit_center'], 
                result['save_dir'],
                result['crop_image'],
                result['square_size'], 
                result['grid_x'], 
                result['grid_y']
    ) 
    
    # Retorna os valores coletados
    return (
        result['linha'],
        result['device_name'],
        result['device_path'],
        result['camera_backend'],
        result['option_visualize'],
        result['perc_top'],
        result['perc_bottom'],
        result['min_score'],
        result['limit_center'],
        result['save_dir'],
        result['crop_image'],
        result['square_size'],
        result['grid_x'],
        result['grid_y']
    )


if __name__ == '__main__':
    # INPUTS
    type_model = 'FRCNN_RN50'
    model = load_model(type_model)
    config_path = r'app\config.txt'
    exposure_value = 0.0
    sec_run_model = 0.4
    wait_key = 16

    # Verificar se a GPU está disponível e configurar o dispositivo
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDispositivo de processamento utilizado: {torch_device}')

    current_directory = os.path.dirname(os.path.abspath(__file__))   # Diretório atual
    parent_directory = os.path.dirname(current_directory)   # Diretório pai (pasta acima)

    # Iniciar a aplicação
    #linha, device_name,, device_path, option_visualize = start_application_without_interface()
    linha, device_name, device_path, camera_backend, option_visualize, perc_top, perc_bottom, \
            min_score, limit_center, save_dir, cropped_image, square_size, grid_x, grid_y = start_application_interface(config_path)

    #print(square_size, grid_x, grid_y)
    # Caso seja uma câmera, converter em número
    try:
        device_path = int(device_path)  # Tenta converter para inteiro
    except ValueError:
        pass

    # Inicia o Device
    (device,device_fps,device_width,device_height,device_exposure) = device_start(device_name, camera_backend, device_path)

    if device:
        if camera_backend == "OpenCV":
            device = device_config(device_name, device, device_fps, device_width, device_height, device_exposure)

        device_start_capture(camera_backend, torch_device, device_name, device, device_fps, type_model, model,
                              option_visualize, sec_run_model, perc_top, perc_bottom, wait_key, config_path, 
                              exposure_value, min_score, limit_center, save_dir, linha, cropped_image, square_size, grid_x, grid_y
        )