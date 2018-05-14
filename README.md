# Installation

``` 
git clone  https://github.com/jijyisme/senior-project-backend.git
pip install -r senior-project-backend/requirements.txt
```

# Start server

```
cd senior-project-backend/aylien
python manage.py runserver
```
the server will listening on port 8000

# File Functionality
- `aylien/`: Contains url and setting files to fit the requirement of Django project.
- `nlp_tools/`
    - `models.py`
        - Structuring and manipulating the data
    - `serializers.py`
        - Translating Django models into text-based formats and used for sending Django data over APIs.
    - `views.py`
        - Encapsulate the logic responsible for processing the API requests and returning the response for each NLP tasks.
    - `__init__.py`, `admin.py`, `apps.py` and `tests.py`
        - Other abstraction layers by Django
