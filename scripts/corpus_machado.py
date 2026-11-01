import glob
import os
import re
import nltk
import zipfile

#para limpar o dataset utilizei praticas conhecidas no card de NLP onde foi apresentado o REGEX como uma das opções para limpeza de texto

nltk.download('machado')
# download da base 

zip_path = os.path.expanduser('~/nltk_data/corpora/machado.zip')
    
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(os.path.join('./machado_data', 'corpora'))
    # extrai o zip com a base

fileids = glob.glob(os.path.join('./machado_data', 'corpora', 'machado', '**', '*.txt'), recursive=True)
# caminhos para cada txt dentro do zip

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    # remove qualquer sequencia de espaços
    return text.strip()

def remove_header(raw):
    # essa funcao remove boa parte dos cabecalhos
    pattern = re.compile(
        r"^.*?Texto-fonte:.*?\d{4}\.\s*\n+"
        # seleciona do inicio ate a linha de edicao (ano + ponto) logo apos "Texto-fonte:"
        r"(?:Publicado originalmente.*?\d{2}/\d{2}/\d{4}\s*\n+)?",
        # pega a publicação presente no header quando ela existe, pois em algumas linhas nao tem
        re.DOTALL
    )
    return pattern.sub("", raw, count=1)
#substituindo somente o header(1 ocorrencia)

with open('machado.txt', "w", encoding="utf-8") as out_file:
    for path in fileids:
        #para cada arquivo le o txt
        with open(path, 'r', encoding='cp1252') as p:
            lin = p.read()
        #aplica a funcao de remover o header
        lin = remove_header(lin)
        #funcao de limpar o texto
        txt = clean_text(lin)
        if txt:
            # se tiver texto vai escrever no txt como output
            out_file.write(txt)
            out_file.write("\n\n")

