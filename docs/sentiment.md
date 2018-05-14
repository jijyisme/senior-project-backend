# Sentiment

**URL** : `/sentiment/`

**Method** : `POST`

**Auth required** : No

**Permissions required** : None

## Success Response

**Code** : `200 OK`

**Content examples**

```json
{
    "confidence_tag_list": [
      { "tag": "POS", "confidence": 0.6 },
      { "tag": "NEG", "confidence": 0.4 }
    ],
}
```

## Notes

* If the text from input URL cannot be retrieved, return status 400 (BAD_REQUEST)
* If the text from input string is too large (>10000), return status 413 (REQUEST_ENTITY_TOO_LARGE)
