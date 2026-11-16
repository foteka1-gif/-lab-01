
# findex

![CI](https://github.com/foteka1-gif/-lab-01/actions/workflows/ci.yml/badge.svg)




Пошуковий рушій за інвертованим індексом із підтримкою булевих запитів, фразового пошуку, ранжування (BM25, TF-IDF) та CLI-інтерфейсу.

## Встановлення

Проєкт налаштовано для встановлення однією командою через `uv`:

```bash
uv tool install .
```

Або зі зібраного `wheel`-пакета:

```bash
uv tool install findex-0.4.0-py3-none-any.whl
```

## Використання CLI

* **Побудова індексу з папки документів:**
  ```bash
  findex index data/ -o data/index.bin --positions
  ```
* **Ранжований пошук (вивід таблицею або у JSON):**
  ```bash
  findex search data/index.bin "python AND search"
  findex search data/index.bin "ангел" --json
  ```
* **Статистика індексу:**
  ```bash
  findex stats data/index.bin
  ```
* **Діагностичні логи (`stderr`):** прапорці `-v` (INFO) та `-vv` (DEBUG) перед підкомандою.

## Тестування та покриття коду

* Загальне покриття тестами (`pytest --cov=findex`): **98%** (130 тестів, включно з property-based тестами на Hypothesis).
* **Навмисно залишилося поза покриттям:**
  * `src/findex/__main__.py` — мінімальна обгортка точки входу для виклику пакета через `python -m findex`.
  * Захисні гілки `except OSError` у `corpus.py` та `cli.py` — обробка рідкісних помилок доступу до диска чи видалення файлу вже після побудови індексу.
