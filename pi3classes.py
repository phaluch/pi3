from random import choice, shuffle, choices
import copy
from tqdm import tqdm
from math import ceil
import pandas as pd
from datetime import datetime as time
from pi3.graph import *

class Pacote():
    """
    Item com nome, descrição, origem e destino
    """

    def __init__(self, origem, destino, grafo, nome='Nome Generico', desc='Pacote Generico') -> None:
        """
        @param origem: Nó de origem do 
        @param nome: String para identificar pacote. Puramente estético
        @param desc: Descrição do pacote. Puramente estético
        """
        self.nome = nome
        self.desc = desc
        self.origem = origem
        self.destino = destino
        self.grafo = grafo
        self.caminho = grafo.caminho(origem,destino)
        self.hash = str(self.origem) + '$' + str(self.destino)
        self.distancia = grafo.menoresDistancias[origem][destino]

    def __repr__(self) -> str:
        return f'{str(self.origem)} -> {str(self.destino)}'
        


class Caminhao():
    """
    Cromossomo do nosso algoritmo genético
    """

    def __init__(self, pacotes, grafo, capacidade = 50) -> None:
        self.pacotes = pacotes
        self.capacidade = capacidade
        self.grafo = grafo
        curMMC = self.grafo.MMC(list(self.getParadas()),.4)
        self.melhorCaminho = curMMC[1]

    def __str__(self):
        resposta = {'pacotes':self.pacotes,
                    'capacidade':self.capacidade,
                    'melhorCaminho':self.melhorCaminho,
                    'getParadas':self.getParadas(),
                    'getItinerario':self.getItinerario(),
                    'getDistancia':self.getDistancia()}
        return str(resposta)

    def atualizarMelhorCaminho(self):
        # Essa função é chamada aqui, em vez de um método, porque tem um componente aleatório
        paradas = list(self.getParadas())
        if len(paradas) == 0:
            self.melhorCaminho = None
        else:
            curMMC = self.grafo.MMC(paradas,.4)
            self.melhorCaminho = curMMC[1]

    def npacotes(self):
        return len(self.pacotes)

    def espacoLivre(self):
        return self.capacidade - self.npacotes()
        
    def getNumPacotesUnicos(self):
        hashes = [pacote.hash for pacote in self.pacotes]
        hashesUnicos = set(hashes)
        return len(hashesUnicos)

    def getPctPacotesUnicos(self):
        if self.npacotes() == 0:
            return 0
        return self.getNumPacotesUnicos()/self.npacotes() # Este número tem que diminuir


    def getParadas(self):
        """
        Todos os pontos únicos pelos quais o caminhão precisa passar para entregas
        """
        paradas = set()
        for pacote in self.pacotes:
            paradas.add(pacote.origem)
            paradas.add(pacote.destino)
        return paradas

    def getItinerario(self):
        if self.melhorCaminho is None:
            return None
        return self.grafo.itinerario(self.melhorCaminho)

    def getDistancia(self):

        if self.getItinerario() is None:
            return 0
        total = 0
        for i in range(len(self.getItinerario())-1):
            x, y = self.getItinerario()[i], self.getItinerario()[i+1]
            total += self.grafo.menoresDistancias[x][y]
        return total # Este número tem que diminuir


class ConjuntoDeCaminhoes():
    def __init__(self, frota, pacotes) -> None:
        self.pacotes = pacotes
        if len(frota) < 1:
            raise ValueError('A lista não pode estar vazia.')
        self.frota = frota
        
    def getDistanciaTotal(self):
        return sum([caminhao.getDistancia() for caminhao in self.frota.values()])
        #self.distanciaTotal = sum([caminhao.getDistancia() for caminhao in self.frota.values()])

    def adicionarPacoteAleatoriamente(self, pacote):
        caminhaoAleatorio = choice(self.frota.keys())
        caminhaoAleatorio.append(pacote)
        caminhaoAleatorio.atualizarMelhorCaminho()

class AGPI3():
    def __init__(self) -> None:
        self.fitnessGeracoes = {}        
        self.curGeracao = 0

    def gerarGrafo(self, vertices, arestas, intervalo=(1,10)):
        self.grafo = Grafo()
        self.grafo.popularAutomaticamente(vertices, arestas, intervalo)
        self.grafo.preencherMenoresCaminhos()

    def gerarPacotes(self, nPacotes=40, origem = None):
        pacotes = []
        for _ in range(nPacotes):
            
            if origem is None:
                origem = choice(list(self.grafo.vertices()))
            destino = choice(list(self.grafo.vertices()))
            curPacote = Pacote(origem, destino, self.grafo)
            pacotes.append(curPacote)
        return pacotes

    def gerarCaminhao(self, pacotes, capacidade = 50):
        caminhao = Caminhao(pacotes,self.grafo, capacidade = 50)
        return caminhao

    def gerarIndividuo(self, n_caminhoes, pacotes = None, origem = None, capacidade = 50):
        
        if type(pacotes) == int:
            pacotes = self.gerarPacotes(nPacotes=pacotes, origem=origem)
        elif type(pacotes) != list:
            raise(f'O tipo {type(pacotes)} não é aceito para a variável pacotes. Esperado: int, list')
        individuo = {i:self.gerarCaminhao(pacotes[i::n_caminhoes], capacidade) for i in range(n_caminhoes)}
        return ConjuntoDeCaminhoes(individuo, pacotes)

    def gerarPopulacao(self, n_individuos, n_caminhoes, pacotes = None, origem=None, capacidade = 50):
        pop = []
        for _ in tqdm(range(n_individuos)):
            pop.append(self.gerarIndividuo(n_caminhoes,pacotes=pacotes,origem=origem, capacidade = capacidade))
        self.pop = pop
        # Reset desses valores ao gerar nova população, pra poder usar o mesmo conjunto de pacotes.
        self.fitnessGeracoes = {}        
        self.curGeracao = 0


    # CLASSES DE TREINAMENTO
    def fitness(self, ind):
        # Para o caminhao -> diminuir o percentual único
        # Para o individuo -> diminuir a distanciaTotal
        
        somaPctUnico = sum([caminhao.getPctPacotesUnicos() for caminhao in ind.frota.values()])
        nCaminhoes = len(ind.frota)
        somaPonderada = somaPctUnico/nCaminhoes

        listaParadas = [caminhao.getParadas() for caminhao in ind.frota.values() if caminhao.getParadas()]
        intersection = set.intersection(*listaParadas)
        union = set.union(*listaParadas)
        pctIntersection = len(intersection)/len(union)
        result = ind.getDistanciaTotal() * (1 + somaPonderada/5 + pctIntersection/5)
        #print(f'{ind.getDistanciaTotal():.2f} * (1 + {somaPonderada:.2f} + {pctIntersection:.2f}) = {result:.2f}')
        #return ind.getDistanciaTotal()
        return result

    def crossover(self, caminhao1, caminhao2, origem = None):
        for p1 in caminhao1.getParadas():
            if p1 == origem:
                continue
            if p1 in caminhao2.getParadas():
                for i,pacote in enumerate(caminhao2.pacotes):
                    if caminhao1.espacoLivre() < 1:
                        break
                    if (pacote.origem == p1) or (pacote.destino == p1):
                        troca = caminhao2.pacotes.pop(i)
                        caminhao1.pacotes.append(troca)
        #print(f'Caminhao 1:')
        #print(caminhao1)
        caminhao1.atualizarMelhorCaminho()
        #print(f'Caminhao 2:')
        #print(caminhao2)
        caminhao2.atualizarMelhorCaminho()

    def mutacao(self, caminhao):
        if caminhao.melhorCaminho is not None:
            if len(caminhao.melhorCaminho) > 3:    
                comeco = caminhao.melhorCaminho[0]
                fim = caminhao.melhorCaminho[-1]
                to_shuffle = caminhao.melhorCaminho[1:-1]
                shuffle(to_shuffle)
                caminhao.melhorCaminho = [comeco] + to_shuffle + [fim]


    def individuoMutacao(self, cjDeRotas):
        #print([cam.melhorCaminho for cam in cjDeRotas.frota.values()])
        start = time.now()
        cur = copy.deepcopy(cjDeRotas)
        end = time.now()
        print('>>[MUTACAO]cur = copy.deepcopy(cjDeRotas) ',end-start)
        start = end
        #print([cam.melhorCaminho for cam in cur.frota.values()])
        chosenIndex = choice([x for x in cur.frota])
        self.mutacao(cur.frota[chosenIndex])
        end = time.now()
        print('>>[MUTACAO]self.mutacao(cur.frota[chosenIndex]) ',end-start)
        start = end
        #print([cam.melhorCaminho for cam in cur.frota.values()])
        return cur

    def individuoCrossover(self, cjDeRotas, origem = None):
        #print([cam.getParadas() for cam in cjDeRotas.frota.values()])
        start = time.now()
        cur = copy.deepcopy(cjDeRotas)
        end = time.now()
        print('>>[CROSSOVER]cur = copy.deepcopy(cjDeRotas) ',end-start)
        start = end
        #print([cam.getParadas() for cam in cur.frota.values()])
        chosenIndex1 = choice([x for x in cur.frota])
        chosenIndex2 = choice([x for x in cur.frota])
        while chosenIndex1 == chosenIndex2:
            chosenIndex2 = choice([x for x in cur.frota])
        end = time.now()
        print('>>[CROSSOVER]while chosenIndex1 == chosenIndex2: ',end-start)
        start = end
        self.crossover(cur.frota[chosenIndex1],cur.frota[chosenIndex2],origem)
        end = time.now()
        print('>>[CROSSOVER]self.crossover(cur.frota[chose...: ',end-start)
        start = end
        #print([cam.getParadas() for cam in cur.frota.values()])
        return cur

    def proxGeracao(self, origem = None):
        start = time.now()
        curPop = self.pop        
        n_individuos = len(self.pop)
        
        if len(self.fitnessGeracoes) == 0:
            # Esse if é pra preencher o fitness da geração 0 antes de começar
            fitnessInicial = [self.fitness(ind) for ind in curPop]
            self.fitnessGeracoes[self.curGeracao] = fitnessInicial
            self.curGeracao += 1
        end = time.now()
        print('if len(self.fitnessGeracoes) == 0:',end-start)
        start = end
        
        # Parte das mutações
        mutacoesPop = [self.individuoMutacao(ind) for ind in curPop]
        end = time.now()
        print('mutacoesPop = [self.individuoMutacao(ind) for ind in curPop] ',end-start)
        start = end
        # Parte dos crossover
        crossoverPop = [self.individuoCrossover(ind, origem) for ind in curPop]
        end = time.now()
        print('crossoverPop = [self.individuoCrossover(ind, origem) for ind in curPop] ',end-start)
        start = end
        conjuntoAtual = curPop + mutacoesPop + crossoverPop

        # Pegando os valores de fitness
        fitnessConjuntoAtual = [ceil(self.fitness(ind)) for ind in conjuntoAtual]
        print(len(fitnessConjuntoAtual))
        weights = [(1+abs(max(fitnessConjuntoAtual)-fitness))**3 for fitness in fitnessConjuntoAtual] # O 1+ é para evitar que fique tudo 0 quando ficarem iguais.
        end = time.now()
        print('weights = [(1+abs(max(fit... ',end-start)
        start = end
        
        #print('weights: ',weights)
        #print('fit: ',fitnessConjuntoAtual)
        #print('#',self.curGeracao)
        
        newPop = [choices(conjuntoAtual,weights=weights,k=1)[0] for _ in range(n_individuos)]
        fitnessAtual = [self.fitness(ind) for ind in newPop]
        #print('fitNewPop: ',fitnessAtual)
        self.fitnessGeracoes[self.curGeracao] = fitnessAtual
        self.curGeracao += 1
        end = time.now()
        #print('self.curGeracao += 1 ',end-start)
        start = end
        
        self.pop = newPop
        
    def Evoluir(self, n_geracoes=None, max_geracoes=None, min_variacao=None, strikes = 3, testNum = 10, origem = None):
        """
        Dois modos:
            Passando n_geracoes -> roda n_geracoes vezes, sem distinção.
            Passando max_geracoes + min_variacao -> Para assim que  OU houver variação no valor mínimo do fitness entre gerações MENOR que min_variação,
                                                                    OU houver max_geracoes iteracoes

            A preferência é dada para o primeiro método; Se n_geracoes é passado, os outros atributos são ignorados.
        """
        start = time.now()
        if n_geracoes is not None:
            for _ in tqdm(range(n_geracoes)):
                self.proxGeracao(origem)
                #print(min(self.fitnessGeracoes[self.curGeracao-1]))
            end = time.now()
            print('if n_geracoes is not None:',end-start)
            start = end
        else:
            if (max_geracoes is None) or (min_variacao is None):
                raise Exception(f'Como você não passou n_geracoes, max_geracoes e min_variacao não podem ser nulos.')
            strike = 0
            #for _ in tqdm(range(max_geracoes)):
            
            for _ in range(max_geracoes):
                self.proxGeracao(origem)
                end = time.now()
                print('self.proxGeracao(origem):',end-start)
                start = end
                #print(self.curGeracao)
                #print(self.fitnessGeracoes.keys())
                # Essas atribuições podem ser feitas porque a primeira chamada de proxGeracao preenche os valores pras gerações 0 e 1
                for ind in self.pop:
                    #print([x.getParadas() for x in ind.frota.values()])
                    pass
                
                if self.curGeracao > testNum:
                    valorAnterior = sum(self.fitnessGeracoes[self.curGeracao-testNum])/len(self.fitnessGeracoes[self.curGeracao-testNum])
                    valorAtual = sum(self.fitnessGeracoes[self.curGeracao-1])/len(self.fitnessGeracoes[self.curGeracao-1])
                    #print(f'valorAnterior: {valorAnterior}; valorAtual: {valorAtual}')
                    variacao = abs(1- abs(valorAtual/valorAnterior))
                    end = time.now()
                    print('if self.curGeracao > testNum:',end-start)
                    start = end
                else:
                    variacao = 1
                #print('Var: ',variacao)
                #print(f'Strike: {strike}')
                if variacao < min_variacao:
                    strike += 1
                    if strike > strikes:
                        break
                else:
                    strike = 0
