#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#define MAX_AMOSTRAS 1000       // Máximo de amostras
#define CARACTERISTICAS 8       // Características por amostra

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

// Função para encontrar os k vizinhos mais próximos
void encontraKVizinhosMaisProximos(Vizinho vizinhos[], int k, float *pontoDeTeste, float xTrain[][CARACTERISTICAS], int tamanhoDoTreino) {
    for (int i = 0; i < tamanhoDoTreino; i++) {
        vizinhos[i].distancia = distanciaEuclidiana(pontoDeTeste, xTrain[i], CARACTERISTICAS);
        vizinhos[i].indice = i;
    }

    // Ordenar os vizinhos por distância
    for (int i = 0; i < tamanhoDoTreino - 1; i++) {
        for (int j = 0; j < tamanhoDoTreino - i - 1; j++) {
            if (vizinhos[j].distancia > vizinhos[j + 1].distancia) {
                Vizinho temp = vizinhos[j];
                vizinhos[j] = vizinhos[j + 1];
                vizinhos[j + 1] = temp;
            }
        }
    }
}

// Função para encontrar os k vizinhos mais próximos paralelizada
void encontraKVizinhosMaisProximosParalelo(Vizinho vizinhos[], int k, float *pontoDeTeste, float xTrain[][CARACTERISTICAS], int tamanhoDoTreino) {
    int i, j;

#pragma omp parallel for private(j) shared(vizinhos, pontoDeTeste, xTrain)
    for (i = 0; i < tamanhoDoTreino; i++) {
        vizinhos[i].distancia = distanciaEuclidiana(pontoDeTeste, xTrain[i], CARACTERISTICAS);
        vizinhos[i].indice = i;
    }

    // Ordenação paralelizada
#pragma omp parallel for private(j)
    for (i = 0; i < tamanhoDoTreino - 1; i++) {
        for (j = 0; j < tamanhoDoTreino - i - 1; j++) {
            if (vizinhos[j].distancia > vizinhos[j + 1].distancia) {
#pragma omp critical
                {
                    Vizinho temp = vizinhos[j];
                    vizinhos[j] = vizinhos[j + 1];
                    vizinhos[j + 1] = temp;
                }
            }
        }
    }
}

// Função para realizar a votação das classes
float votar(Vizinho vizinhos[], int k, float *yTrain) {
    float votos = 0;

    for (int i = 0; i < k; i++) {
        votos += yTrain[vizinhos[i].indice];
    }

    return (votos > (k / 2)) ? 1.0 : 0.0; // Supondo que as classes são 0 e 1
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
    float precisao;

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

    precisao = ((float)predicoesCorretas / tamanhoDoTeste) * 100.0;
    printf("\nPrecisao do Modelo KNN com k=%d: %.2f%%\n", k, precisao);
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
    float xTrain[MAX_AMOSTRAS][CARACTERISTICAS], yTrain[MAX_AMOSTRAS];
    float xTest[MAX_AMOSTRAS][CARACTERISTICAS], yTest[MAX_AMOSTRAS];
    float predicoes[MAX_AMOSTRAS];
    int trainSize, testSize; // Definindo k = 3 para o KNN

    // Lendo os dados de treinamento e teste
    lerDadosEixoX("xtrain.txt", xTrain, &trainSize);
    lerDadosEixoY("ytrain.txt", yTrain, trainSize);
    lerDadosEixoX("xtest.txt", xTest, &testSize);

    double startTime, endTime;

    for (int k = 1; k <= 10; k++) {
        startTime = omp_get_wtime();
        testKNN(xTrain, yTrain, trainSize, xTest, yTest, testSize, k, false, predicoes, knnOriginal);
        endTime = omp_get_wtime();
        printf("Tempo KNN Original com k=%d: %f segundos\n", k, endTime - startTime);

        startTime = omp_get_wtime();
        testKNN(xTrain, yTrain, trainSize, xTest, yTest, testSize, k, false, predicoes, knnParalelo);
        endTime = omp_get_wtime();
        printf("Tempo KNN Paralelo com k=%d: %f segundos\n", k, endTime - startTime);
    }

    escreverPredicoes("ytest.txt", predicoes, testSize);
}
