
Project Organization
------------

    ├── LICENSE
    ├── README.md                       <- The top-level README for developers using this project.
    ├── requirements.txt 
    ├── src                             <- Source code for use in this project.
    │   ├── utils
    │   └── zero_shot_classification
    │       ├── classes.txt             <- All classes used for classificator.
    │       └── model.py                <- The Zero-Shot-Classificator model.
    └── example.ipynb                   <- Jupyter notebook example.


--------
## Example
```python
from src.zero_shot_classification.model import Classificator
model = Classificator(device="cuda")
model.init_classes(["Python", "Excel", "программа 1С", "Маркетинг", "Machine learning"])

pred = model.predict(
    text="Приглашаем на работу специалиста в отдел логистики. Организация грузоперевозок до магазинов сети. Требования: "
         "Опыт работы логистом от 1 года; Умение работать с большим объемом информации; Знание особенностей маршрутов и"
         "направлений грузоперевозок; Коммуникабельность, ответственность внимательность, стрессоустойчивость; Уверенный "
         "пользователь ПК (программа 1С);",
    thresh=0.5
)
```

## Result
```python
['программа 1С']
```