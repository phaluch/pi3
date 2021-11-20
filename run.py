from pi3.pi3classes import *

origem = 0
capacidade = 11
paiDeTodos = AGPI3()
paiDeTodos.gerarGrafo(vertices=400,arestas=0.4)
pacotes = paiDeTodos.gerarPacotes(100,origem)
n_individuos = 15
n_caminhoes = 10
max_geracoes = 100
recebedorDeResultados = []

paiDeTodos.gerarPopulacao(n_individuos=n_individuos,n_caminhoes=n_caminhoes,pacotes=pacotes, capacidade = capacidade)
paiDeTodos.Evoluir(max_geracoes=max_geracoes, min_variacao=0.005, strikes = 10, origem = origem)