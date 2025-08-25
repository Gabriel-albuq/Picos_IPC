import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox

def load_settings(config_path):
    """Função para ler as configurações de um arquivo .txt e atribuir os valores diretamente às variáveis."""
    # Valores padrão
    defaults = {
        'linha': 'Linha',
        'device_name': 'Camera',
        'device_path': '0',
        'camera_backend': 'OpenCV',
        'option_visualize': 1,
        'perc_top': 0.4,
        'perc_bottom': 0.8,
        'perc_median': 0.3,
        'min_score': 0.5,
        'limit_center': 8,
        'save_dir': 'data\\outputs\\capturas',
        'salvar_maiores': 0,
        'salvar_menores': 1000,
        'deslocamento_esquerda': 780,
        'deslocamento_direita': 280,
        'box_size': 540,
        'box_distance': 820,
        'box_offset_x': -120
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

    if defaults['save_dir'] == 'None':
        defaults['save_dir'] = None

    return (
        defaults['linha'],
        defaults['device_name'],
        defaults['device_path'],
        defaults['camera_backend'],
        defaults['option_visualize'],
        defaults['perc_top'],
        defaults['perc_bottom'],
        defaults['perc_median'],
        defaults['min_score'],
        defaults['limit_center'],
        defaults['save_dir'],
        defaults['salvar_maiores'],
        defaults['salvar_menores'],
        defaults['deslocamento_esquerda'],
        defaults['deslocamento_direita'],
        defaults['box_size'],
        defaults['box_distance'],
        defaults['box_offset_x'],
    )


def save_settings(config_path, linha, device_name, device_path, camera_backend, option_visualize, perc_top, 
                 perc_bottom, perc_median, min_score, limit_center, 
                 save_dir, salvar_maiores, salvar_menores, deslocamento_esquerda, deslocamento_direita,
                 box_size, box_distance, box_offset_x):
    """Função para salvar as configurações atuais diretamente nas variáveis no arquivo .txt."""
    arquivo = config_path
    try:
        with open(arquivo, 'w') as file:
            file.write(f'linha = {linha}\n')
            file.write(f'device_name = {device_name}\n')
            file.write(f'device_path = {device_path}\n')
            file.write(f'camera_backend = {camera_backend}\n')
            file.write(f'option_visualize = {option_visualize}\n')
            file.write(f'perc_top = {perc_top}\n')
            file.write(f'perc_bottom = {perc_bottom}\n')
            file.write(f'perc_median = {perc_median}\n')
            file.write(f'min_score = {min_score}\n')
            file.write(f'limit_center = {limit_center}\n')
            if save_dir is not None:
                file.write(f'save_dir = {save_dir}\n')
            else:
                file.write(f'save_dir = None\n')
            file.write(f'salvar_maiores = {salvar_maiores}\n')
            file.write(f'salvar_menores = {salvar_menores}\n')
            file.write(f'deslocamento_esquerda = {deslocamento_esquerda}\n')
            file.write(f'deslocamento_direita = {deslocamento_direita}\n')
            file.write(f'box_size = {box_size}\n')
            file.write(f'box_distance = {box_distance}\n')
            file.write(f'box_offset_x = {box_offset_x}\n')
            
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
        'perc_median': None,
        'min_score': None,
        'limit_center': None,
        'save_dir': None,
        'salvar_maiores': None,
        'salvar_menores': None,
        'camera_backend': None, 
        'deslocamento_esquerda': None,
        'deslocamento_direita': None,
        'box_size': None,
        'box_distance': None,
        'box_offset_x': None,
    }

    (
        linha,
        device_name,
        device_path,
        camera_backend,
        option_visualize,
        perc_top,
        perc_bottom,
        perc_median,
        min_score,
        limit_center,
        save_dir,
        salvar_maiores,
        salvar_menores,
        deslocamento_esquerda,
        deslocamento_direita,
        box_size,
        box_distance,
        box_offset_x,
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
        if not perc_top_entry.get():
            messagebox.showerror("Erro", "O campo 'Percentual Mínimo' não pode estar vazio.")
            return
        if not perc_median_entry.get():
            messagebox.showerror("Erro", "O campo 'Percentual da Mediana' não pode estar vazio.")
        if not perc_bottom_entry.get():
            messagebox.showerror("Erro", "O campo 'Percentual Máximo' não pode estar vazio.")
            return
        if not min_score_entry.get():
            messagebox.showerror("Erro", "O campo 'Score Mínimo' não pode estar vazio.")
            return
        if not limit_center_entry.get():
            messagebox.showerror("Erro", "O campo 'Limite de centro' não pode estar vazio.")
            return
        if not desloc_esq_entry.get():
            messagebox.showerror("Erro", "O campo 'Deslocamento Esquerda' não pode estar vazio.")
            return
        if not desloc_dir_entry.get():
            messagebox.showerror("Erro", "O campo 'Deslocamento Direita' não pode estar vazio.")
            return
        if not box_size_entry.get():
            messagebox.showerror("Erro", "O campo 'Tamanho da Caixa' não pode estar vazio.")
            return
        if not box_distance_entry.get():
            messagebox.showerror("Erro", "O campo 'Distância entre Caixas' não pode estar vazio.")
            return
        if not box_offset_x_entry.get():
            messagebox.showerror("Erro", "O campo 'Offset Horizontal da Caixa' não pode estar vazio.")
            return

        # Preenche o dicionário result com os valores dos campos
        result['linha'] = linha_entry.get()
        result['device_name'] = device_name_entry.get()
        result['device_path'] = device_path_var.get()
        result['option_visualize'] = int(option_var.get())
        result['perc_top'] = float(perc_top_entry.get())
        result['perc_bottom'] = float(perc_bottom_entry.get())
        result['perc_median'] = float(perc_median_entry.get())
        result['min_score'] = float(min_score_entry.get())
        result['limit_center'] = int(limit_center_entry.get())
        result['camera_backend'] = camera_backend_var.get()
        result['deslocamento_esquerda'] = int(desloc_esq_entry.get())
        result['deslocamento_direita'] = int(desloc_dir_entry.get())
        result['box_size'] = int(box_size_entry.get())
        result['box_distance'] = int(box_distance_entry.get())
        result['box_offset_x'] = int(box_offset_x_entry.get())
        
        # Verifica a opção de salvar detecções
        if not save_detection_var.get():  # Caso "Não salvar detecções" esteja marcado
            result['save_dir'] = save_dir_var.get()
        else:
            result['save_dir'] = None
            
        result['salvar_maiores'] = salvar_maiores
        result['salvar_menores'] = salvar_menores
        
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

    row = -1
    
    # Entrada para a linha
    row+=1
    tk.Label(root, text="Digite a linha:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    linha_entry = tk.Entry(root, width=30)
    linha_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    
    # Entrada para o nome da câmera/vídeo
    row+=1
    tk.Label(root, text="Nome da Câmera/Vídeo:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    device_name_entry = tk.Entry(root, width=30)
    device_name_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    
    # Entrada para o caminho do dispositivo
    row+=1
    tk.Label(root, text="Câmera/Vídeo:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    device_path_var = tk.StringVar()
    device_path_entry = tk.Entry(root, textvariable=device_path_var, width=30)
    device_path_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    
    # Botão para procurar um arquivo de vídeo
    tk.Button(root, text="Procurar", command=browse_file, width=10).grid(row=row, column=2, padx=pad_x, pady=pad_y)
    
    # Seleção de backend da câmera
    row+=1
    tk.Label(root, text="Backend da Câmera:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    camera_backend_var = tk.StringVar(value="")
    frame_backend = tk.Frame(root)
    frame_backend.grid(row=row, column=1, padx=pad_x, pady=pad_y, sticky='w')
    tk.Radiobutton(frame_backend, text="OpenCV", variable=camera_backend_var, value="OpenCV").pack(side='left')
    tk.Radiobutton(frame_backend, text="GxCam", variable=camera_backend_var, value="GxCam").pack(side='left')
    
    # Opção de visualização
    row+=1
    tk.Label(root, text="Visualizar predições:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    option_var = tk.StringVar(value=1)
    frame_options = tk.Frame(root)
    frame_options.grid(row=row, column=1, padx=pad_x, pady=pad_y, sticky='w')
    tk.Radiobutton(frame_options, text="Sim", variable=option_var, value="1").pack(side='left')
    tk.Radiobutton(frame_options, text="Não", variable=option_var, value="0").pack(side='left')
    
    # Diretório de salvar
    row+=1
    tk.Label(root, text="Salvar detecções em:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    save_dir_var = tk.StringVar()
    save_dir_entry = tk.Entry(root, textvariable=save_dir_var, width=30)
    save_dir_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)

    save_dir_var.set(save_dir)  # Preenche com o valor do config.txt
    tk.Button(root, text="Procurar", command=browse_save_dir, width=10).grid(row=row, column=2, padx=pad_x, pady=pad_y)

    # Caminho padrão abaixo do campo
    row+=1
    tk.Label(root, anchor='w', width=30).grid(row=row, column=1, padx=pad_x, pady=pad_y, sticky='w')

    # Opção para "Não salvar detecções" (agora com valor padrão marcado)
    save_detection_var = tk.BooleanVar(value=True)  # Agora inicia como True (marcado)
    save_detection_checkbox = tk.Checkbutton(root, text="Não salvar detecções", variable=save_detection_var, command=toggle_save_dir)
    save_detection_checkbox.grid(row=row, column=1, padx=pad_x, pady=pad_y, sticky='w')

    # Parâmetros adicionais
    # Percentual mínimo
    row+=1
    tk.Label(root, text="Percentual Trigger Superior:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    perc_top_entry = tk.Entry(root, width=30)
    perc_top_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    perc_top_entry.insert(0, perc_top)  # Preenche com o valor do config.txt
    
    # Percentual máximo
    row+=1
    tk.Label(root, text="Percentual Trigger Inferior    :", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    perc_bottom_entry = tk.Entry(root, width=30)
    perc_bottom_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    perc_bottom_entry.insert(0, perc_bottom)  # Preenche com o valor do config.txt

    # Percentual mediana
    row+=1
    tk.Label(root, text="Percentual da Mediana    :", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    perc_median_entry = tk.Entry(root, width=30)
    perc_median_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    perc_median_entry.insert(0, perc_median)  # Preenche com o valor do config.txt
    
    # Score mínimo
    row+=1
    tk.Label(root, text="Score Mínimo:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    min_score_entry = tk.Entry(root, width=30)
    min_score_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    min_score_entry.insert(0, min_score)  # Preenche com o valor do config.txt
    
    # Limite de centro
    row+=1
    tk.Label(root, text="Limite de centro:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    limit_center_entry = tk.Entry(root, width=30)
    limit_center_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    limit_center_entry.insert(0, limit_center)  # Preenche com o valor do config.txt

    # Deslocamento esquerda
    row+=1
    tk.Label(root, text="Deslocamento Esquerda:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    desloc_esq_entry = tk.Entry(root, width=30)
    desloc_esq_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    desloc_esq_entry.insert(0, deslocamento_esquerda) 

    # Deslocamento direita
    row+=1
    tk.Label(root, text="Deslocamento Direita:", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    desloc_dir_entry = tk.Entry(root, width=30)
    desloc_dir_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    desloc_dir_entry.insert(0, deslocamento_direita) 

    # Tamanho da caixa (box_size)
    row+=1
    tk.Label(root, text="Tamanho da Caixa (box_size):", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    box_size_entry = tk.Entry(root, width=30)
    box_size_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    box_size_entry.insert(0, box_size) 

    # Distância da caixa (box_distance)
    row+=1
    tk.Label(root, text="Distância entre Caixas (box_distance):", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    box_distance_entry = tk.Entry(root, width=30)
    box_distance_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    box_distance_entry.insert(0, box_distance) 

    # Offset X da caixa (box_offset_x)
    row+=1
    tk.Label(root, text="Offset Horizontal (box_offset_x):", anchor='w', width=30).grid(row=row, column=0, padx=pad_x, pady=pad_y, sticky='w')
    box_offset_x_entry = tk.Entry(root, width=30)
    box_offset_x_entry.grid(row=row, column=1, padx=pad_x, pady=pad_y)
    box_offset_x_entry.insert(0, box_offset_x) 
        
    # Botão de confirmação
    row+=1
    tk.Button(root, text="Confirmar", command=submit, width=20).grid(row=row, column=0, columnspan=4, pady=10)
    
    # Chama a função para ajustar o estado do diretório de salvar
    toggle_save_dir()
    
    root.mainloop()
    
    # Salva as configurações
    save_settings(config_path,       
                    result['linha'],
                    result['device_name'],
                    result['device_path'],
                    result['camera_backend'],
                    result['option_visualize'],
                    result['perc_top'],
                    result['perc_bottom'],
                    result['perc_median'],
                    result['min_score'],
                    result['limit_center'],
                    result['save_dir'],
                    result['salvar_maiores'],
                    result['salvar_menores'],
                    result['deslocamento_esquerda'],
                    result['deslocamento_direita'],
                    result['box_size'],
                    result['box_distance'],
                    result['box_offset_x'],
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
        result['perc_median'],
        result['min_score'],
        result['limit_center'],
        result['save_dir'],
        result['salvar_maiores'],
        result['salvar_menores'],
        result['deslocamento_esquerda'],
        result['deslocamento_direita'],
        result['box_size'],
        result['box_distance'],
        result['box_offset_x']
    )


if __name__ == '__main__':
    # Exemplo de chamada da função
    config_path = r'app\config.txt'
    (linha, device_name, device_path, camera_backend, option_visualize, exposure_value, 
     perc_top, perc_bottom, min_score, limit_center, save_dir, 
     sec_run_model, wait_key, square_size, grid_x, grid_y) = start_application_interface(config_path)
    # print("Linha:", linha)
    # print("Nome da Câmera/Vídeo:", device_name)
    # print("Caminho do dispositivo:", device_path)
    # print("Visualizar predições:", option_visualize)
    # print("Exposição:", exposure_value)
    # print("Percentual Mínimo:", perc_top)
    # print("Percentual Máximo:", perc_bottom)
    # print("Score Mínimo:", min_score)
    # print("Limite de centro:", limit_center)
    # print("Tamanho da área:", square_size)
    # print("Localização X:", grid_x)
    # print("Localização Y:", grid_y)
    # print("Diretório de salvar:", save_dir if save_dir else "Não salvar detecções")