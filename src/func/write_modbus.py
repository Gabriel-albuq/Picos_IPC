import time
from modbus import read_modbus, write_modbus

try:
    while True:
        #IP para CLP Codesys
        IP_LEITURA="10.107.241.128" 
        IP_ESCRITA="10.107.241.128"

        VAR_ESCRITA=[25,10,30,50]

        # LEITURA=read_modbus(IP_LEITURA)
        # print(LEITURA)
        # time.sleep(0.5)

        ESCRITA=write_modbus(IP_ESCRITA, VAR_ESCRITA)
        print(ESCRITA)
        time.sleep(0.5)
except:
    print("FALHA NO APP")
