from pyModbusTCP.client import ModbusClient

def read_modbus(IP):
    #Configura comunicação com o CLP
    comm_config = ModbusClient(host=IP, port=502, unit_id=1, timeout=500.0, auto_open=True, auto_close=True)
    #Executa a leitura dos registros no CLP
    regs_read=comm_config.read_input_registers(0, 10)

    #Verifica se a leitura dos dados foi realizada com sucesso
    if regs_read:
        #Após leitura, transforma os volores inteiros em valores reais conforme unidade de engeharia
        i=0
        while i < 6:
            regs_read[i]=regs_read[i]/100
            i+=1 #i=i+1 
    else:
        regs_read="Erro na leitura"

    return regs_read
   
def write_modbus(IP, vars_escrita):
    #Configura comunicação com o CLP
    comm_config = ModbusClient(host=IP, port=502, unit_id=1, timeout=500.0, auto_open=True, auto_close=True)
    #Executa a escrita dos registros no CLP
    reg_write=comm_config.write_multiple_registers(0, vars_escrita)

        #Verifica se a escrita dos dados foi realizada com sucesso
    if reg_write:
        reg_write="Escrita finalizada com sucesso"
    else:
        reg_write="Erro na escrita"
    return reg_write








