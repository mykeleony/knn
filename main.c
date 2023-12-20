#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#define MAX_AMOSTRAS 1000000
#define CARACTERISTICAS 8          // Características por amostra

// Estrutura para armazenar a distância e o índice de um ponto de treinamento
typedef struct {
    float distancia;
    int indice;
} Vizinho;

// Função para ler os dados de x (características)
void lerDadosEixoX(const char *nomeDoArquivo, float dadosDeX[][CARACTERISTICAS], int *tamanho) {
    FILE *arquivo = fopen(nomeDoArquivo, "r");

    if (arquivo == NULL) {
        perror("Erro ao abrir o arquivo");
        exit(EXIT_FAILURE);
    }

    float valor;
    char separador;
    int i = 0, j;

    while (fscanf(arquivo, "%f%c", &valor, &separador) == 2 && i < MAX_AMOSTRAS) {
        dadosDeX[i][0] = valor;

        for (j = 1; j < CARACTERISTICAS; ++j) {
            fscanf(arquivo, "%f%c", &dadosDeX[i][j], &separador);
        }

        i++;
    }

    *tamanho = i;
    fclose(arquivo);
}

// Função para ler os dados de y (rótulos)
void lerDadosEixoY(const char *nomeDoArquivo, float dadosDeY[], int tamanho) {
    FILE *arquivo = fopen(nomeDoArquivo, "r");

    if (arquivo == NULL) {
        perror("Erro ao abrir o arquivo");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tamanho; ++i) {
        fscanf(arquivo, "%f", &dadosDeY[i]);
    }

    fclose(arquivo);
}

// Função para calcular a distância euclidiana
float distanciaEuclidiana(float *a, float *b, int tamanho) {
    float soma = 0.0;

    for (int i = 0; i < tamanho; i++) {
        soma += pow(a[i] - b[i], 2);
    }

    return sqrt(soma);
}

// Função de comparação para qsort
int compararVizinhos(const void *a, const void *b) {
    Vizinho *vizinhoA = (Vizinho *)a;
    Vizinho *vizinhoB = (Vizinho *)b;

    if (vizinhoA->distancia < vizinhoB->distancia) return -1;
    if (vizinhoA->distancia > vizinhoB->distancia) return 1;
    return 0;
}

// Função para encontrar os k vizinhos mais próximos
void encontraKVizinhosMaisProximos(Vizinho vizinhos[], int k, float *pontoDeTeste, float xTrain[][CARACTERISTICAS], int tamanhoDoTreino) {
    for (int i = 0; i < tamanhoDoTreino; i++) {
        vizinhos[i].distancia = distanciaEuclidiana(pontoDeTeste, xTrain[i], CARACTERISTICAS);
        vizinhos[i].indice = i;
    }

    // Ordenar os vizinhos por distância usando qsort
    qsort(vizinhos, tamanhoDoTreino, sizeof(Vizinho), compararVizinhos);
}

void encontraKVizinhosMaisProximosParalelo(Vizinho vizinhos[], int k, float *pontoDeTeste, float xTrain[][CARACTERISTICAS], int tamanhoDoTreino) {
    // Paralelizando o cálculo da distância
#pragma omp parallel for
    for (int i = 0; i < tamanhoDoTreino; i++) {
        vizinhos[i].distancia = distanciaEuclidiana(pontoDeTeste, xTrain[i], CARACTERISTICAS);
        vizinhos[i].indice = i;
    }

    // Usando qsort para ordenação paralela
#pragma omp parallel
    {
#pragma omp single nowait
        {
            qsort(vizinhos, tamanhoDoTreino, sizeof(Vizinho), compararVizinhos);
        }
    }
}

// Função para realizar a votação das classes
float votar(Vizinho vizinhos[], int k, float *yTrain) {
    float votos = 0;

    for (int i = 0; i < k; i++) {
        votos += yTrain[vizinhos[i].indice];
    }

    return (votos > (k / 2)) ? 1.0 : 0.0; // Classes 0 e 1
}

float knn(float xTrain[][CARACTERISTICAS], float *yTrain, float *xTest, int tamanhoDoTreino, int k) {
    Vizinho vizinhos[tamanhoDoTreino];
    encontraKVizinhosMaisProximos(vizinhos, k, xTest, xTrain, tamanhoDoTreino);

    return votar(vizinhos, k, yTrain);
}

float knnParalelo(float xTrain[][CARACTERISTICAS], float *yTrain, float *xTest, int tamanhoDoTreino, int k) {
    Vizinho vizinhos[tamanhoDoTreino];
    encontraKVizinhosMaisProximosParalelo(vizinhos, k, xTest, xTrain, tamanhoDoTreino);

    return votar(vizinhos, k, yTrain);
}

void testKNN(float xTrain[][CARACTERISTICAS], float yTrain[], int tamanhoDoTreino, float xTest[][CARACTERISTICAS], float yTest[], int tamanhoDoTeste, int k, bool flagDetalhado, float predicoes[]) {
    int predicoesCorretas = 0;
    double inicio = omp_get_wtime(); // Início da marcação de tempo

    for (int i = 0; i < tamanhoDoTeste; i++) {
        float predito = knn(xTrain, yTrain, xTest[i], tamanhoDoTreino, k);
        predicoes[i] = predito; // Armazenando a predição

        if (flagDetalhado) {
            printf("Amostra %d: Classe Real = %f, Classe Predita = %f\n", i + 1, yTest[i], predito);
        }

        if (predito == yTest[i]) {
            predicoesCorretas++;
        }
    }

    double fim = omp_get_wtime(); // Fim da marcação de tempo
    printf("Tempo de execucao (KNN Normal) para %d amostras: %f segundos\n", tamanhoDoTreino, (fim - inicio));
}

void testKNNParalelo(float xTrain[][CARACTERISTICAS], float yTrain[], int tamanhoDoTreino, float xTest[][CARACTERISTICAS], float yTest[], int tamanhoDoTeste, int k, bool flagDetalhado, float predicoes[]) {
    int predicoesCorretas = 0;
    double inicio = omp_get_wtime(); // Início da marcação de tempo

    for (int i = 0; i < tamanhoDoTeste; i++) {
        float predito = knnParalelo(xTrain, yTrain, xTest[i], tamanhoDoTreino, k);
        predicoes[i] = predito; // Armazenando a predição

        if (flagDetalhado) {
            printf("Amostra %d: Classe Real = %f, Classe Predita = %f\n", i + 1, yTest[i], predito);
        }

        if (predito == yTest[i]) {
            predicoesCorretas++;
        }
    }

    double fim = omp_get_wtime();
    printf("Tempo de execucao (KNN Paralelo) para %d amostras: %f segundos\n", tamanhoDoTreino, (fim - inicio));
}

// Função para escrever as predições em um arquivo
void escreverPredicoes(const char *nomeDoArquivo, float predicoes[], int tamanhoDoTeste) {
    FILE *arquivo = fopen(nomeDoArquivo, "w");

    if (arquivo == NULL) {
        perror("Erro ao abrir o arquivo para escrita");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tamanhoDoTeste; i++) {
        fprintf(arquivo, "%.1f\n", predicoes[i]);
    }

    fclose(arquivo);
}

int main() {
    float (*xTrain)[CARACTERISTICAS] = malloc(MAX_AMOSTRAS * sizeof(*xTrain));
    float *yTrain = malloc(MAX_AMOSTRAS * sizeof(*yTrain));
    float (*xTest)[CARACTERISTICAS] = malloc(MAX_AMOSTRAS * sizeof(*xTest));
    float *yTest = malloc(MAX_AMOSTRAS * sizeof(*yTest));
    float *predicoes = malloc(MAX_AMOSTRAS * sizeof(*predicoes));

    if (!xTrain || !yTrain || !xTest || !yTest || !predicoes) {
        fprintf(stderr, "Falha na alocação de memória.\n");

        free(xTrain);
        free(yTrain);
        free(xTest);
        free(yTest);
        free(predicoes);
        exit(EXIT_FAILURE);
    }

    int trainSize, testSize;
    int k = 3;

    const char *datasets[] = {"100", "500", "1000", "5000", "10000", "20000", "50000", "100000", "200000", "500000"};
    int numDatasets = sizeof(datasets) / sizeof(datasets[0]);

    for (int d = 0; d < numDatasets; d++) {
        char xTrainFile[50], yTrainFile[50], xTestFile[50] = "xtest.txt", yTestFile[50];

        // Gerar nomes de arquivo
        sprintf(xTrainFile, "xtrain%s.txt", datasets[d]);
        sprintf(yTrainFile, "ytrain%s.txt", datasets[d]);
//        sprintf(xTestFile, "xtest%s.txt", datasets[d]);
        sprintf(yTestFile, "ytest%s.txt", datasets[d]);

        // Lendo os dados de treinamento e teste
        lerDadosEixoX(xTrainFile, xTrain, &trainSize);
        lerDadosEixoY(yTrainFile, yTrain, trainSize);
        lerDadosEixoX(xTestFile, xTest, &testSize);

        // Executando testes com KNN Normal e Paralelo
        testKNN(xTrain, yTrain, trainSize, xTest, yTest, testSize, k, false, predicoes);
        testKNNParalelo(xTrain, yTrain, trainSize, xTest, yTest, testSize, k, false, predicoes);

        // Escrevendo as predições em um arquivo
        escreverPredicoes(yTestFile, predicoes, testSize);
    }

    free(xTrain);
    free(yTrain);
    free(xTest);
    free(yTest);
    free(predicoes);

    return 0;
}
