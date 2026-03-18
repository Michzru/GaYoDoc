# Python Virtual Environment Setup

This project uses a Python virtual environment (`venv`) to manage dependencies.

---

## Create Virtual Environment

Run the following command in the root directory of the project:

```bash
python -m venv venv
```

This will create a `venv/` directory containing the isolated environment.

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

After activation, the terminal should display `(venv)`.

---

## Install Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Deactivate Environment

To deactivate the virtual environment:

```bash
deactivate
```

