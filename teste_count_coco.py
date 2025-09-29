import json
from collections import defaultdict

# --- ALTERE AQUI ---
# Coloque o caminho completo para o seu arquivo JSON.
# Exemplo para Windows: 'C:/Users/SeuUsuario/Documentos/anotacoes.json'
# Exemplo para Linux/Mac: '/home/seu_usuario/documentos/anotacoes.json'
caminho_do_arquivo = r'data/inputs/train_images/COCO_20250908/valid/_annotations.coco.json'
# -----------------

try:
    # Abre o arquivo JSON em modo de leitura ('r') com codificação UTF-8
    with open(caminho_do_arquivo, 'r', encoding='utf-8') as f:
        # 1. Carregar os dados diretamente do arquivo
        data = json.load(f)

    # 2. Verificar se a chave 'annotations' existe
    if 'annotations' not in data:
        raise KeyError("O arquivo JSON não contém a chave 'annotations', que é necessária para contar os objetos.")

    # 3. Contar objetos por image_id
    object_counts = defaultdict(int)
    for annotation in data['annotations']:
        object_counts[annotation['image_id']] += 1

    # 4. Criar uma lista para armazenar os resultados (nome_do_arquivo, contagem)
    image_details = []
    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']
        count = object_counts[image_id]
        image_details.append((file_name, count))

    # 5. Ordenar a lista pela contagem, do menor para o maior
    image_details_sorted = sorted(image_details, key=lambda item: item[1])

    # 6. Imprimir os resultados formatados
    print("Contagem de objetos por imagem (ordenado do menor para o maior):\n")
    for file_name, count in image_details_sorted:
        print(f"Imagem: {file_name} - Objetos: {count}")

except FileNotFoundError:
    print(f"Erro: O arquivo não foi encontrado no caminho especificado: '{caminho_do_arquivo}'")
    print("Por favor, verifique se o caminho e o nome do arquivo estão corretos.")
except json.JSONDecodeError:
    print(f"Erro: O arquivo '{caminho_do_arquivo}' não contém um JSON válido. Verifique a formatação do arquivo.")
except KeyError as e:
    print(f"Erro: {e}")