# Автоматическая разметка данных

Проект предназначен для автоматической разметки данных для решения задач детектирования объектов. 
Для детектирования используется нейронная сеть [Grounding DINO](https://arxiv.org/pdf/2303.05499.pdf?ref=blog.roboflow.com). Вдохновением для создания ноутбука послужила статья [Grounding DINO:SOTA Zero-Shot Object Detection](https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/) и [туториал](https://github.com/roboflow/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb?ref=blog.roboflow.com) в формате jupyter ноутбука.

В данном проекте реализованы:
1. Запуск Docker контейнера с предустановленными зависимостями; 
2. Установка классов и порогов детектирования;
3. Детектирование объектов;
4. Сохранение разметки в формате YOLO;
5. Создание архива для загрузки разметки в CVAT.

Разметка сохраняется в формате YOLO и упаковывается в архив, который можно загрузить в инструмент разметки данных [CVAT](https://www.cvat.ai/).

![automated-dataset-annotation.jpg](docs%2Fautomated-dataset-annotation.jpg)

### Установка
Для запуска проекта необходимо установить [docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) с поддержкой nvidia.

### Запуск
Через docker-compose:
``` bash
docker-compose up -d --build
```

Через docker:
``` bash
docker build -t automated-dataset-annotation:v1 .
docker run --name automated-dataset-annotation \
           --gpus all \
           --ipc=host \
           --network=host -it --rm \
           -e JUPYTER_TOKEN='automated-dataset-annotation' \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/notebooks:/app/notebooks \
           -v $(pwd)/src:/app/src \
           automated-dataset-annotation:v1
```

После выполнения команды запустится jupyter lab, он доступен по ссылке:
``` 
http://localhost:8888/?token=automated-dataset-annotation
```

### Подготовка данных
Перед началом работы необходимо расположить нужные данные в директории **data**.

### Загрузка разметки в CVAT
Инструкция приведена в файле - [CVAT](docs%2FCVAT.md).

### Описание структуры проекта
```
├── data                                        <- Директория для хранения данных
│
├── docs                                        <- Дополнительнрые материалы
│
├── notebooks                                   <- Jupyter notebooks
│   └── pdn-automated-dataset-annotation        <- Автоматическая разметка данных
│
├── src
│   ├── utils.py                                <- Вспомогательные функции
│   └── worker.py                               <- Запуск нейронной сети
│
├── requirements.txt                            <- Зависимости, необходимые для запуска проекта
│
├── Dockerfile                                  <- Файл для сборки Docker образа
│
└── docker-compose.yml                          <- Файл для запуска сервиса
```
