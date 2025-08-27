from pylogix import PLC

# Endereço IP do CLP

CLP_IP = '192.168.1.10'  # Substitua pelo IP do seu CLP

# Variáveis a serem lidas e escritas

variaveis_para_ler = ['count_cookies_right', 'count_cookies_left']

variaveis_para_escrever = {

    'count_cookies_right': total_detections_right,

    'count_cookies_left': total_detections_left

}

# Conexão com o CLP

with PLC() as comm:

    comm.IPAddress = CLP_IP

    # print("🔍 Lendo variáveis:")

    # for tag in variaveis_para_ler:

    #     resultado = comm.Read(tag)

    #     if resultado.Status == 'Success':

    #         print(f"{tag}: {resultado.Value}")

    #     else:

    #         print(f"Erro ao ler {tag}: {resultado.Status}")

 

    print("\n✍️ Escrevendo variáveis:")

    for tag, valor in variaveis_para_escrever.items():

        resultado = comm.Write(tag, valor)

        if resultado.Status == 'Success':

            print(f"{tag} atualizado para {valor}")

        else:

            print(f"Erro ao escrever em {tag}: {resultado.Status}")