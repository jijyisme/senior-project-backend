# Backend Website 
Backend using Django for Thai_NLP_Platform (also see (github)[https://github.com/KawinL/Thai_NLP_platform]) as a senior project for Computer Engineering, Chulalongkorn University (CP41)

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

## RESTful documentation

* [Tokenization](docs/tokenize.md) : `POST /tokenize/`
* [Word Embedding](docs/vectorize.md) : `POST /vectorize/`
* [Part-of-speech Tagging](docs/pos.md) : `POST /pos/`
* [Named Entity Recognition](docs/ner.md) : `POST /ner/`
* [Sentiment](docs/sentiment.md) : `POST /sentiment/`
* [Categorization](docs/categorization.md) : `POST /categorization/`
* [Keyword Expansion](docs/keyword_expansion.md) : `POST /keyword_expansion/`
* [Get vector distance (for word embedding)](docs/vector_distance.md) : `POST /vector_distance/`
* [Get tag lists](docs/model_taglist.md) : `POST /model_taglist/`

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
