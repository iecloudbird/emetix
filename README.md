# jobhedge-investor

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/iecloudbird/jobhedge-investor.git
   cd jobhedge-investor
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Development

- Always activate the virtual environment before working on the project
- To add new dependencies, install them and update requirements.txt:
  ```bash
  pip install package_name
  pip freeze > requirements.txt
  ```

### Deactivating the virtual environment

```bash
deactivate
```
