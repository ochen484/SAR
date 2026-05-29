 # versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel


## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all', 'positional',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.positional = False # valor por defecto, se cambia con self.index_dir() dependiendo de los argumentos de entrada

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        Hector 

        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        #1 - 
        nltk.download('punkt')
        nltk.download('punkt_tab')
        frases = nltk.sent_tokenize(txt)

        #2 - 
        for frase in frases:
            indice = len(self.chuncks)
            self.chuncks.append(frase) #lista plana de todas las frases de todos los articulos, el indice de cada frase es su posicion en esta lista
            self.chunck_index.append(indice) #Registra pos
            self.artid_to_emb[indice] = artid #mapea el indice del chunk con el artid del articulo al que peretence. Se usara en el KDTree  

                     
        

   #Oscar
    
    def create_kdtree(self):
        """
        Crea el KDTree utilizando el modelo semántico cargado.
        Solo se debe llamar una vez, tras indexar todos los documentos.

        Pasos:
            1 - Cargar el modelo semántico (si no está cargado)
            2 - Llamar al método fit del modelo con la lista de chuncks
            3 - Guardar el kdtree y los embeddings generados en el indexer
        """
        print(f"Creating kdtree ...", end="")

        
        self.load_semantic_model()
        if self.model is None:
            print("ERROR, no se cargo el modelo", file=sys.stderr)
            return

       
        self.model.fit(self.chuncks)

       
        self.kdtree = self.model.kdtree       
        self.embeddings = self.model.embeddings  
        print("done!")


    #Oscar    
    def solve_semantic_query(self, query:str): 
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()
        if self.model is None:
            return [] 
       
        top_k = self.MAX_EMBEDDINGS  #Nº max de resultados a recuperar del KDTree, puede aumentar si no hay resultados imp
        total_chunks = len(self.chuncks)   
        results = self.model.query(query, top_k)
        while (self.semantic_threshold is not None #Tenemos umbral semntico definido
            and len(results) > 0 #Hay resultados
            and results[-1][0] <= self.semantic_threshold #Los resultados no superan el umbral semantico, es decir, el último resultado tiene una distancia menor o igual al umbral, por lo tanto no se han recuperado todos los resultados relevantes
            and top_k < total_chunks): #Aun podriamos recuperar mas resultados porque no hemos llegado al máximo de embeddings disponibles
            top_k = min(top_k * 2, total_chunks) #Como todavia quedan resultados por recuperar doblamos el top_k (capado a total_chunks)
            results = self.model.query(query, top_k) #Una vez doblada la variable relanzamos la consulta para recuperar mas resultados  


        seen = set() #Variable local para evitar repetir articulos  
        articles = []

        for distance, chunk_idx in results: #distance y chunk_idx son elementos de la tupla q devuelve query. Distance: dtca semantica entre la consulta y el chunk, chunk_idx: indice del chunk en self.chuncks
         if self.semantic_threshold is not None and distance > self.semantic_threshold: #Si nos hemos pasado del umbral semantic paramos, no nos sirven los results
            break
         artid = self.artid_to_emb[chunk_idx] #mapea el indice del chunk con el artid del articulo al que peretence. 
         if artid not in seen:
            seen.add(artid)
            articles.append(artid)

        return articles

    
    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()
        if self.model is None:
            return articles  # no se pudo cargar el modelo, devolvemos el ranking original sin cambios
       
        top_k = self.MAX_EMBEDDINGS
        total_chunks = len(self.chuncks)
        articles_set = set(articles)  
     
        

        seen = set()
        ranked_articles = []
        results = self.model.query(query, top_k) #lista de (dtca y chunk_idx) ordenada de menor a mayor
            
        while len(seen) < len(articles_set) and top_k < total_chunks:
            #Resetear los vistos y el ranking para cada iteración. Consultamos la iteración anterior (en el for)
            seen = set()
            ranked_articles = []
            #tupla (distance, chunk_idx)
            for _, chunk_idx in results:
                artid = self.artid_to_emb[chunk_idx]
                if artid in articles_set and artid not in seen:
                    seen.add(artid)
                    ranked_articles.append(artid)
            top_k = min(top_k * 2, total_chunks)
            results = self.model.query(query, top_k)

        remaining = []
        for a in articles:
            if a not in seen:
                remaining.append(a)

        return ranked_articles + remaining
    

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)
        
        if self.semantic:
            self.create_kdtree()

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article

    #Oscar
    def index_file(self, filename: str):
    
    #Indexa el contenido de un fichero.
    #input: "filename" es el nombre de un fichero generado por el Crawler;
     #      cada línea es un objeto json con la información de un artículo de Wikipedia.

    #NECESARIO PARA TODAS LAS VERSIONES

    #dependiendo del valor de self.positional se debe ampliar el indexado
    

    # Asignar un docid único a este fichero si no lo tiene aún
     docid = len(self.docs)
     self.docs[docid] = filename

     for i, line in enumerate(open(filename)):
        j = self.parse_article(line)

        # Saltar artículos con URL duplicada
        if self.already_in_index(j):
            continue

        # Registrar la URL para evitar duplicados
        self.urls.add(j['url'])

        artid = len(self.articles)

        # Guardar info del artículo
        self.articles[artid] = (docid, i, j['url'], j['title']) #i = position en fichero. Esto se usara en solve_and_show para mostrar la info al usuario

        tokens = self.tokenize(j[self.DEFAULT_FIELD])

        if self.positional: #Esto será util para get_positionals, para recuperar los artículos que contienen una secuencia de términos consecutivos
            # Índice posicional: posting list con listas de posiciones
            for pos, token in enumerate(tokens): #Enumerate hace que devuelva pares (pos,token)
                if token not in self.index:
                    self.index[token] = {}
                if artid not in self.index[token]:
                    self.index[token][artid] = []
                self.index[token][artid].append(pos)
        else: #NO guarda las posiciones, solo los artid en una lista
            seen = set()
            for token in tokens:
                if token not in self.index:
                    self.index[token] = []
                if artid not in seen:
                    self.index[token].append(artid)
                    seen.add(artid)

        if self.semantic:
            self.update_chuncks(j[self.DEFAULT_FIELD], artid) #DEFAULT_FILED decide que elemnto es relevante para la indexacion (URL, title, summary, all).

    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        Hector
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        print("=" * 40)
        print("Cantidad de archivos indexados:", len(self.docs))
        print("Cantidad de artículos indexados:", len(self.articles))
        print("Cantidad de términos indexados:", len(self.index))

        if self.positional:
            total_postings = sum(len(v) for v in self.index.values())
            print("Cantidad de total postings:", total_postings)

        if self.semantic:
           print("Cantidad de chunks (frases):", len(self.chuncks))

        print("=" * 40)



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        Hector
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        
        if query is None or len(query) == 0:
         return [], query  # resultado vacío para consulta vacía

        result = None  

        tokens = re.findall(r'"[^"]+"|\S+', query)

        i = 0
        while i < len(tokens):
         negate = False

         # Detectar NOT
         if tokens[i].upper() == 'NOT':
            negate = True
            i += 1
            if i >= len(tokens): 
                break

         token = tokens[i]

         if token.startswith('"') and token.endswith('"'):
            # Búsqueda posicional
            terms = self.tokenize(token[1:-1])  # quitar comillas y tokenizar
            posting = self.get_positionals(terms)
         else:
            # Término simple
            posting = self.get_posting(token)

         if negate:
            posting = self.reverse_posting(posting)

         if result is None:
            result = posting
         else:
            result = self.and_posting(result, posting)

         i += 1

        return (result if result is not None else []), query #Este caso cubre la posibilidad de que la query sea solo "NOT". 
        #En el caso de solo "NOT" se activa el flag "negate" pero como no hay más tokens hace break y result sigue siendo None, por eso devolvemos la lista vacía []




    def get_posting(self, term:str):
        """
        Hector
        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        term = term.lower()
        if self.positional:
             #Si el termino esta en el indice, devuelve la lista de artids ordenada. Si no esta devuelve una lista vacia.
             return sorted(self.index[term].keys()) if term in self.index else [] #.keys() devuelve los artids (id articulos)
            #Sorted ordena las .keys(), es decir, los artids, de forma creciente. De menor a mayor. 
            #Que este ordenada importa para metodos como and_posting y/o  reverse_posting
        else:
             #self.index ya es una lista de artids ordenada, si el termino esta en el indice devuelve esa lista, si no esta devuelve una lista vacia.
             return self.index.get(term, [])


    #Oscar
    def get_positionals(self, terms: list):
    
    #Devuelve la posting list asociada a una secuencia de términos consecutivos.
    #NECESARIO PARA LAS BÚSQUEDAS POSICIONALES

    #param:  "terms": lista con los términos consecutivos para recuperar la posting list.

   # return: posting list (lista ordenada de artids que contienen los términos consecutivos)
    

     if not terms:
        return []

    # Obtener el índice posicional del primer término
     if terms[0] not in self.index:
        return []

     
     current = dict(self.index[terms[0]])   #Creamos una copia del dicc para no mod el original. Se ira reduciendo con cada iteracion

     for term in terms[1:]: #2do elemento en adelante   
        if term not in self.index:
            return [] #Si el termino no esta en el indice no existe la secuencia completa, por lo tanto devolvemos lista vacia

        next_term_index = self.index[term] #Guardo el termino siguiente para comparar con el actual
        new_current = {} #Dicc dnd guardo solo aquellos que superen el filtro de consecutividad, se ira actualizando con cada iteracion

        for artid, positions in current.items(): #artid, pos es una tupla 2 elementos, artid es el id del articulo y positions es la lista de posiciones del termino actual en ese articulo
            if artid not in next_term_index:
                continue

            consecutive_positions = [] #array que guarda las pos consecutivas del siguiente termino, se va actualizando con cada iteracion
            pos_next = next_term_index[artid] #las pos dnd aparace el term. Evitara acceder al ind original, una copia local

            i, k = 0, 0
            while i < len(positions) and k < len(pos_next):
                expected = positions[i] + 1 #La posición esperada del siguiente término es la posición actual + 1, porque deben ser consecutivos
                if pos_next[k] == expected: #Coinciden, se añade a la lista de posiciones consecutivas y se avanza en ambos
                    consecutive_positions.append(pos_next[k])
                    i += 1
                    k += 1
                elif pos_next[k] < expected: #Aqui pues si nos hemos quedao cortos avanzamos k 
                    k += 1
                else: #Si nos hemos pasado avanzamos i
                    i += 1

            if consecutive_positions: #comprueba que hay elementos consecutivos y los añade al nuevo dicc. Si no el artid se descarta
                new_current[artid] = consecutive_positions

        current = new_current #actualizamos el las posiciones actuales con las consecutivas del siguiente término, se irá reduciendo con cada iteracion

     return sorted(current.keys()) #Posting list ordenadas por artids   




    def reverse_posting(self, p:list):
        """
        Hector
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        all_articles = sorted(self.articles.keys())
        result = []
        i, j = 0, 0 #i para avanzar en all_articles, j para avanzar en p
        while i < len(all_articles):
         if j < len(p) and all_articles[i] == p[j]: #si no me he pasado de len y el artid actual es igual al de p[j], es decir, el artid esta en p.
            j += 1   #Entonces no hago nada y avanzo pq este metodo es como un NOT
         else:
            result.append(all_articles[i]) #El artid actual no esta en p, me sirve por ende lo añado
         i += 1
        return result



    def and_posting(self, p1:list, p2:list):
        """
        Hector
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        #Estamos suponiendo posting lists ordenadas
        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
         if p1[i] == p2[j]:
            result.append(p1[i]) #son iguales, avanzo ambos y añado a resultado
            i += 1
            j += 1
         elif p1[i] < p2[j]: #como p1[i] es menor avanzo i mantengo j 
            i += 1
         else:
            j += 1 #Viceversa
        return result






    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors

  
    def solve_and_show(self, query:str):
        """
        Hector
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
          # Obtener la posting list
        
        result, _ = self.solve_query(query) #descartamos la query procesada que devuelve solve_query, solo nos interesa la posting list result
        if result is None:
            result = [] #en teoria no deberia pasar pq solve_query devuelve [] si no hay resultados, pero por si acaso lo ponemos para evitar errores al hacer len(result) mas adelante
        total = len(result)

        to_show = result if self.show_all else result[:self.SHOW_MAX] #Si show_all es True mostramos todos los resultados, si no mostramos solo los primeros SHOW_MAX resultados

        print(f"Query: {query}")
        print(f"Number of results: {total}") #aunque no se muestren todos si que incluimos el nº total

        for order, artid in enumerate(to_show, start=1): #Enumerate nos devuelve pares (order, artid) donde empieza en 1 (start=1) y aumenta 1 por cada elemento de to_show
           _, _, url, title = self.articles[artid] #Esto es una tupla de 4 elementos (docid, pos, url, title) y nos quedamos con el url y el title para mostrar al usuario
           #Es lo que viene de index_file
           print(f"  {order}\t{artid}\t{title}\t{url}") #\t son columnas

        return total
        """
        Por ejemplo con to_show = [5, 23, 47]:
            (1, 5) (order, artid) = (1, 5) => muestra el artículo con artid 5 y lo ordena como 1
            (2, 23)
            (3, 47) 
        
        """



