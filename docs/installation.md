# 🛠️ Instalação

## 📋 Pré-requisitos

```text
asttokens==3.0.0  
backcall==0.2.0  
certifi==2025.1.31  
charset-normalizer==3.4.1  
coloredlogs==15.0.1  
comm==0.2.2  
contourpy==1.1.1  
cycler==0.12.1  
Cython==0.29.37  
debugpy==1.8.13  
decorator==5.2.1  
executing==2.2.0  
filelock==3.16.1  
flatbuffers==25.2.10  
fonttools==4.56.0  
fsspec==2025.2.0  
-e aux_install/camera/Galaxy_Linux_Python_2.0.2106.9041/api  
humanfriendly==10.0  
idna==3.10  
importlib_metadata==8.5.0  
importlib_resources==6.4.5  
ipykernel==6.29.5  
ipython==8.12.3  
jedi==0.19.2  
Jinja2==3.1.6  
jupyter_client==8.6.3  
jupyter_core==5.7.2  
kiwisolver==1.4.7  
MarkupSafe==2.1.5  
matplotlib==3.7.5  
matplotlib-inline==0.1.7  
mpmath==1.3.0  
nest-asyncio==1.6.0  
networkx==3.1  
numpy==1.24.4  
nvidia-pyindex==1.0.9  
onnx==1.17.0  
onnxruntime-gpu @ file:///home/mdb/ProjetosPython/Picos/aux_install/files_whl/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl  
opencv-python==4.11.0.86  
packaging==24.2  
parso==0.8.4  
pexpect==4.9.0  
pickleshare==0.7.5  
pillow==10.4.0  
platformdirs==4.3.6  
prompt_toolkit==3.0.50  
protobuf==5.29.3  
psutil==7.0.0  
ptyprocess==0.7.0  
pure_eval==0.2.3  
pycocotools==2.0.7  
Pygments==2.19.1  
pyparsing==3.1.4  
python-dateutil==2.9.0.post0  
pyzmq==26.3.0  
requests==2.32.3  
six==1.17.0  
stack-data==0.6.3  
sympy==1.13.3  
torch @ file:///home/mdb/ProjetosPython/Picos/aux_install/files_whl/torch-2.1.0a0%2B41361538.nv23.06-cp38-cp38-linux_aarch64.whl  
torchvision @ file:///home/mdb/ProjetosPython/Picos/aux_install/files_whl/torchvision-0.16.1-cp38-cp38-linux_aarch64.whl  
tornado==6.4.2  
traitlets==5.14.3  
typing_extensions==4.12.2  
urllib3==2.2.3  
wcwidth==0.2.13  
zipp==3.20.2  
```

## 🔽 Instalando as Dependências (IPC)

1. **Instalação do Pyenv**

    Instalar as dependências necessárias
    ```bash
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
    ```

    Instalando o Pyenv
    ```bash
    curl https://pyenv.run | bash
    ```

    Adicionando o pyenv ao shell
    ```bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```

    Recarregando o terminal
    ```bash
    source ~/.bashrc   # ou ~/.zshrc, dependendo do seu shell
    ```

3. **Escolher versão do python**

    Entre no diretório do projeto
    ```bash
    cd /caminho/do/seu/projeto
    ```

    Defina a versão do Python a ser utilizada
    ```bash
    pyenv local 3.8.20
    ```

4. **Criar ambiente virtual**

    Dentro do diretório do projeto, digite
    ```bash
    python -m venv .venv
    ```

    Ative o ambiente
    ```bash
    source .venv/bin/activate
    ```

2. **Alterar requirements.txt**

    As bibliotecas gx, torch, torchvision e onnx são bibliotecas não listas no Pypi, são instaladas a partir de arquivos locais. Aletere os caminhos dessas biliotecas para o caminho dos arquivos que estão em seu computador.

3. **Instalar bibliotecas**

    ```bash
    pip install -r requirements.txt
    ```