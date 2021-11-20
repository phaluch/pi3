from collections import defaultdict
from random import randint, choice, random
from tqdm import tqdm
from itertools import permutations

class Grafo():
    """
    Classe genérica para grafos, com métodos como menor caminho entre pontos A e B
    """

    def __init__(self) -> None:
        self.adj = defaultdict(dict)
        self.densidade = None


    def contarArestas(self, adj):
        return sum([len(adj[v]) for v in adj]) / 2 # Divisão por dois porque cada aresta é contada duas vezes (ida e volta)

    def vertices(self):
        return set(self.adj.keys())

    def popularAutomaticamente(self, vertices, arestas, intervalo=(1,10)):
        """
        Cria um grafo não-direcional com os vertices e arestas.
        @param vertices: int, str ou list;
            Traz os vértices que serão usados no grafo
                        int -> Os vértices serão gerados por range(vertices)
                        str -> serão usados os caracteres da string.
                                -> Aceita apenas a-zA-Z, e descarta valores duplicados.
                        list -> serão usados os elementos da lista. Descarta valores duplicados.
                                -> Devem ser hasable items, para serem transformados em set.
                        set -> Os elementos do set serão a lista
        @param arestas: float, 0 < x <= 1; 
            Representa a densidade de arestas a serem geradas automaticamente. 
            O número real será o primeiro valor possível >= ao valor informado.
            O grafo será completamente ligado, então o número mínimo de arestas é vertices-1, e o máximo é vertices!
        @param intervalo: tupla no formato (min, max) (inclusivos) para gerar aleatoriamente as distâncias entre os vértices.
        """

        if len(self.adj) != 0:
            raise Exception('O grafo já está populado. Use .limparGrafo() para apagar todo o conteúdo deste objeto.')
        
        # Transformando os vértices em set, para poder iterar sobre;
        # apesar de virarem as chaves de um dict depois, garantir que sejam hashable e remover duplicidades é bom.
        if type(vertices) == int:
            vertices = set([x for x in range(vertices)])
        elif type(vertices) == str:
            vertices = set([x for x in vertices])
        elif type(vertices) == list:
            vertices = set(vertices)
        elif type(vertices) == set:
            pass
        else:
            raise TypeError(f'Variável do tipo {type(vertices)} não é aceita. Esperava int, str ou list')

        # Dict com as listas vazias
        adj = {v:{} for v in vertices}

        # Populando até a densidade
        maxArestas = sum(range(len(vertices)))/2 # Porque como é não-direcional, serão contadas duas vezes
        minArestas = len(vertices)-1
        minPct = minArestas/maxArestas
        if minPct > arestas:
            arestas = minPct # Garantindo que o grafo será completo, quando usado com o sorted abaixo.
        #print(f'maxArestas: {maxArestas}')
        curPct = self.contarArestas(adj)/maxArestas
        #print(f'{curPct} < {arestas}')

        # Fazendo a população inicial A->B->....->N, para garantir que o grafo é conectado
        verticesSequenciais = list(vertices)
        for i in range(len(vertices)-1):
            distancia = randint(*intervalo)
            adj[verticesSequenciais[i]][verticesSequenciais[i+1]] = distancia
            adj[verticesSequenciais[i+1]][verticesSequenciais[i]] = distancia

        while curPct < arestas:
            # Escolhendo o primeiro dos valores

            A = sorted(vertices, key = lambda x: len(adj[x]))[0] # Pego um dos que tem menos, para distribuir melhor
            if len(adj[A].keys()) == len(vertices):
                A = sorted(vertices, key = lambda x: len(adj[x]), reverse=True)[0] # Pego um dos que tem menos, para distribuir melhor
            possiveisB = vertices - set(adj[A].keys())
            possiveisB.remove(A)
            B = choice(list(possiveisB))
            distancia = randint(*intervalo)
            adj[A][B] = distancia
            adj[B][A] = distancia
            
            curPct = self.contarArestas(adj)/maxArestas
            #print(f'{curPct} = {self.contarArestas(adj)}/{maxArestas}')
            #pprint(adj)
            #print(f'{curPct} < {arestas} = {curPct < arestas}')
            #input()
        self.adj = adj
            

        #Implementing Dijkstra's Algorithm
    def dijkstra(self, inicial):
        visited = {inicial : 0}
        path = defaultdict(list)

        nodes = set(self.adj.keys()) # set(graph.nodes)

        while nodes:
            minNode = None
            for node in nodes:
                if node in visited:
                    if minNode is None:
                        minNode = node
                    elif visited[node] < visited[minNode]:
                        minNode = node
            if minNode is None:
                break

            nodes.remove(minNode)
            currentWeight = visited[minNode]

            for edge in self.adj[minNode]:
                weight = currentWeight + self.adj[minNode][edge]
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge].append(minNode)
        
        return visited, path

    def preencherMenoresCaminhos(self):
        self.menoresDistancias={}
        self.melhorCaminho={}
        for v in self.vertices():
            self.menoresDistancias[v], self.melhorCaminho[v] = self.dijkstra(v)

    def caminho(self,x,y):
        try:
            cam = []
            while True:
                cam.append(x)
                if x == y:
                    return cam[::-1]
                x = self.melhorCaminho[y][x][0]
                
        except:
            print(self.melhorCaminho[x])
            raise Exception(f'x = {x},y = {y}')

    def MMC(self, paradas, chance = 1.):
        """
        Mínimo Melhor Caminho para passar por todas as paradas
        """
        primeiro = paradas[0]
        outras = paradas[1:]

        perms = permutations(outras)
        distanciaFinal = False
        finalPerm = None
        for perm in perms:
            perm = [primeiro] + list(perm)
            #print(perm)
            distancia = 0
            for i in range(len(perm)-1):
                distancia += self.menoresDistancias[perm[i]][perm[i+1]]
                #print(self.menoresDistancias[perm[i]][perm[i+1]],end='+')
            if not distanciaFinal:
                finalPerm = perm
                distanciaFinal = distancia
            else:
                if random() > chance:
                    return distanciaFinal, finalPerm

                if distancia < distanciaFinal:
                    distanciaFinal = distancia
                    finalPerm = perm
            #print(f'={distancia}')
        return distanciaFinal, finalPerm

    def itinerario(self, rota):
        itinerario = [rota[0]]
        for i in range(len(rota)-1):
            itinerario = itinerario + self.caminho(rota[i],rota[i+1])[1:]
        return itinerario